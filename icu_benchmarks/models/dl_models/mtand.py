"""mTAND (Multi-Time Attention Network) for irregularly sampled time series.

Reference:
    Shukla & Marlin, "Multi-Time Attention Networks for Irregularly Sampled
    Time Series", ICLR 2021.
    Paper: https://openreview.net/pdf?id=4c0J6lwQ4_
    Preprint: https://arxiv.org/abs/2101.10318
    Official code: https://github.com/reml-lab/mTAN
        (``models.py``: ``multiTimeAttention``, ``enc_mtan_rnn``,
         ``dec_mtan_rnn``, ``create_classifier``;
         ``tan_classification.py``: the joint training loop;
         ``utils.py``: ``compute_losses``, ``log_normal_pdf``, ``normal_kl``)

This implements the **mTAND-Full** classifier (the variant the paper reports for
PhysioNet/MIMIC-III mortality, ``tan_classification.py --enc mtan_rnn
--dec mtan_rnn``): a VAE-style multi-time-attention encoder maps the observed
(value, mask) pairs to a posterior over latent states at a set of reference
time points; a GRU+MLP classifier predicts from the latent state, and an
mTAND decoder reconstructs the observations. Training optimizes the official
joint objective ``reconstruction_ELBO + alpha * cross_entropy`` (IWAE bound).

Integration notes for YAIB
--------------------------
The YAIB preprocessing pipeline (see ``icu_benchmarks/data/preprocessor.py``)
emits, for every feature column ``<f>``, a paired binary column
``MissingIndicator_<f>`` (1 == originally missing) and then forward/zero-fills
the value column. We recover the (value, mask) pairs by name (same logic as the
GRU-D baseline) so the layout is dataset/task agnostic. The mask gates the
attention (observed-only) and the reconstruction loss, so the forward-filled
values at unobserved positions never contribute.

The data reaches the model on a regular 1-hour grid of fixed length ``T``
(``hirid/mortality24``: T=25, 0..24h, no padding); observation times are that
grid normalized to ``[0, 1]``. mortality24 has a single label per stay at the
last timestep, so the classifier emits one prediction broadcast across ``T``
(``DLPredictionWrapper.step_fn`` scores only the labeled/last step) and the
reconstruction+KL term is returned as ``aux_loss``, which the wrapper adds to
its (balanced) cross-entropy. To keep the classification head shape-compatible
with the wrapper we use ``k_iwae`` latent samples for the reconstruction ELBO
but a single sample for the classifier logits (default ``k_iwae=1`` reproduces
the official objective exactly).
"""

import logging
import math

import gin
import torch
import torch.nn.functional as F
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

MISSING_PREFIX = "MissingIndicator_"


class multiTimeAttention(nn.Module):
    """Multi-time attention module (mTAND). Faithful port of the official code.

    For each (learned) reference time point it computes attention over the
    observation times from a learned time embedding, then takes a per-feature
    masked weighted average of the input channels.

    Args:
        input_dim: channel dim of ``value``.
        nhidden: output embedding size.
        embed_time: dim of the time embedding (divisible by num_heads).
        num_heads: number of attention heads.
    """

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()
        assert embed_time % num_heads == 0, "embed_time must be divisible by num_heads"
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),  # query projection
                nn.Linear(embed_time, embed_time),  # key projection
                nn.Linear(input_dim * num_heads, nhidden),  # output projection
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        """Scaled dot-product attention with per-feature masking.

        query: (B, h, Lq, d_k), key: (B, h, Lk, d_k), value: (B, 1, Lk, dim),
        mask: (B, 1, Lk, dim) or None. Returns (B, h, Lq, dim).
        """
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B,h,Lq,Lk)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)  # (B,h,Lq,Lk,dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)  # softmax over observation times (Lk)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn  # (B,h,Lq,dim)

    def forward(self, query, key, value, mask=None, dropout=None):
        # query/key: (B, L*, embed_time), value/mask: (B, Lk, dim)
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B,1,Lk,dim)
        value = value.unsqueeze(1)  # (B,1,Lk,dim)
        query, key = [
            layer(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key))
        ]  # each (B, h, L*, embed_time_k)
        x, _ = self.attention(query, key, value, mask, dropout)  # (B,h,Lq,dim)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)  # (B,Lq,h*dim)
        return self.linears[-1](x)  # (B, Lq, nhidden)


class _TimeEmbed(nn.Module):
    """Continuous-time embedding shared interface (learned or fixed sinusoidal).

    Learned (paper default contribution): ``[linear(t), sin(periodic(t))]``.
    Fixed: sinusoidal positional encoding on ``48 * t`` (as in the official code).
    """

    def __init__(self, embed_time: int, learn_emb: bool, freq: float):
        super().__init__()
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.freq = freq
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def forward(self, tt):  # tt: (B, L) -> (B, L, embed_time)
        tt = tt.unsqueeze(-1)
        if self.learn_emb:
            return torch.cat([self.linear(tt), torch.sin(self.periodic(tt))], -1)
        d = self.embed_time
        position = 48.0 * tt
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=tt.device) * -(math.log(self.freq) / d))
        pe = torch.zeros(tt.size(0), tt.size(1), d, device=tt.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class EncMtanRNN(nn.Module):
    """mTAND VAE encoder (``enc_mtan_rnn``): observations -> posterior over z0.

    Returns ``(B, num_ref, 2 * latent_dim)`` = concatenated mean and log-variance
    of the latent state at each reference time point.
    """

    def __init__(self, input_dim, ref_points, latent_dim, nhidden, embed_time, num_heads, learn_emb, freq):
        super().__init__()
        self.dim = input_dim
        self.register_buffer("ref_points", ref_points)
        self.embed = _TimeEmbed(embed_time, learn_emb, freq)
        self.att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2))

    def forward(self, x):
        # x: (B, T, 2D) == concat(values, mask). Mask gates the attention.
        mask = x[:, :, self.dim:]
        mask = torch.cat([mask, mask], 2)  # (B, T, 2D)
        batch = x.size(0)
        obs_tp = self._obs_tp(x)
        key = self.embed(obs_tp)  # (B, T, embed_time)
        query = self.embed(self.ref_points.unsqueeze(0)).expand(batch, -1, -1)  # (B, R, embed_time)
        out = self.att(query, key, x, mask)  # (B, R, nhidden)
        out, _ = self.gru(out)  # (B, R, 2*nhidden)
        return self.hiddens_to_z0(out)  # (B, R, 2*latent)

    def _obs_tp(self, x):
        seq_len = x.size(1)
        denom = max(seq_len - 1, 1)
        return torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1) / denom


class DecMtanRNN(nn.Module):
    """mTAND decoder (``dec_mtan_rnn``): latent z -> reconstructed observations."""

    def __init__(self, input_dim, ref_points, latent_dim, nhidden, embed_time, num_heads, learn_emb, freq):
        super().__init__()
        self.register_buffer("ref_points", ref_points)
        self.embed = _TimeEmbed(embed_time, learn_emb, freq)
        self.att = multiTimeAttention(2 * nhidden, 2 * nhidden, embed_time, num_heads)
        self.gru = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim))

    def forward(self, z, obs_tp):
        # z: (N, R, latent), obs_tp: (N, T). Reconstruct values at observation times.
        out, _ = self.gru(z)  # (N, R, 2*nhidden)
        query = self.embed(obs_tp)  # (N, T, embed_time)
        key = self.embed(self.ref_points.unsqueeze(0)).expand(z.size(0), -1, -1)  # (N, R, embed_time)
        out = self.att(query, key, out)  # (N, T, 2*nhidden)
        return self.z0_to_obs(out)  # (N, T, D)


class CreateClassifier(nn.Module):
    """Latent-state classifier (``create_classifier``): GRU over reference points + MLP."""

    def __init__(self, latent_dim, nhidden, num_classes, static_count=0):
        super().__init__()
        self.gru = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden + static_count, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, num_classes),
        )

    def forward(self, z, static=None):
        _, h = self.gru(z)  # h: (1, B, nhidden)
        summary = h.squeeze(0)  # (B, nhidden)
        if static is not None and static.numel() > 0:
            summary = torch.cat([summary, static], dim=-1)
        return self.classifier(summary)  # (B, num_classes)


@gin.configurable
class MTANDNet(DLPredictionWrapper):
    """mTAND-Full classifier for irregularly sampled clinical time series.

    Args:
        input_size: ``(B, T, F)`` shape tuple from the data loader.
        hidden_dim: encoder/classifier GRU width (``rec_hidden`` in the paper).
        num_classes: number of output classes.
        latent_dim: VAE latent dimension per reference point.
        gen_hidden: decoder GRU width (``gen_hidden``).
        embed_time: continuous-time embedding dimension.
        num_heads: attention heads.
        num_ref_points: number of reference time points (fixed ``[0, 1]`` grid).
        learn_emb: learn the time embedding (paper's core contribution).
        freq: base frequency for the fixed sinusoidal embedding (unused if learn_emb).
        alpha: weight of the classification term in ``recon_ELBO + alpha * CE``.
            Implemented as ``aux_loss = recon_loss / alpha`` so the wrapper's CE
            keeps weight 1 and the overall objective is proportional to the paper's.
        k_iwae: number of importance-weighted latent samples for the recon ELBO.
        std: observation noise std for the Gaussian reconstruction likelihood.
        norm: divide per-sample logpx/KL by the observed-measurement count (official
            ``--norm``, on for both mortality commands) so recon is per-observation scaled.
        feature_names: data column names, used to pair value/mask columns.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden_dim,
        num_classes,
        *args,
        latent_dim: int = 32,
        gen_hidden: int = 50,
        embed_time: int = 128,
        num_heads: int = 1,
        num_ref_points: int = 128,
        learn_emb: bool = True,
        freq: float = 10.0,
        alpha: float = 100.0,
        k_iwae: int = 1,
        std: float = 0.01,
        norm: bool = True,
        kl_anneal: bool = True,
        kl_rate: float = 0.99,
        kl_wait: int = 10,
        feature_names=None,
        static_names=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            gen_hidden=gen_hidden,
            embed_time=embed_time,
            num_heads=num_heads,
            num_ref_points=num_ref_points,
            learn_emb=learn_emb,
            freq=freq,
            alpha=alpha,
            k_iwae=k_iwae,
            std=std,
            norm=norm,
            kl_anneal=kl_anneal,
            kl_rate=kl_rate,
            kl_wait=kl_wait,
            feature_names=feature_names,
            static_names=static_names,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.k_iwae = k_iwae
        self.noise_std = std
        self.norm = norm
        self.kl_anneal = kl_anneal
        self.kl_rate = kl_rate
        self.kl_wait = kl_wait

        num_features = input_size[2]
        value_idx, mask_idx, static_idx = self._build_feature_index(feature_names, num_features, static_names)
        # Buffers so the index mapping is saved/restored and device-moved with the model.
        self.register_buffer("value_idx", torch.tensor(value_idx, dtype=torch.long))
        self.register_buffer("mask_idx", torch.tensor(mask_idx, dtype=torch.long))
        self.register_buffer("static_idx", torch.tensor(static_idx, dtype=torch.long))

        decay_dim = len(value_idx)  # D: number of (value, mask) channels
        static_count = len(static_idx)
        logging.info(
            f"MTANDNet (mTAND-Full): {num_features} input features -> {decay_dim} time-series features "
            f"(value/mask pairs), {static_count} static/extra features."
        )

        ref_points = torch.linspace(0.0, 1.0, num_ref_points)
        self.enc = EncMtanRNN(decay_dim, ref_points, latent_dim, hidden_dim, embed_time, num_heads, learn_emb, freq)
        self.dec = DecMtanRNN(decay_dim, ref_points.clone(), latent_dim, gen_hidden, embed_time, num_heads, learn_emb, freq)
        self.clf = CreateClassifier(latent_dim, hidden_dim, num_classes, static_count=static_count)
        # DLPredictionWrapper.set_metrics inspects `self.logit.out_features`.
        self.logit = self.clf.classifier[-1]

    @staticmethod
    def _build_feature_index(feature_names, num_features, static_names=None):
        """Pair each value column with its ``MissingIndicator_<col>`` mask column.

        Returns three index lists into the input feature axis: value indices,
        matching mask indices, and static indices (routed to the classifier branch).

        The YAIB preprocessor applies a ``MissingIndicator`` to *both* dynamic and
        static columns, so a static column also has a ``MissingIndicator_<col>``
        partner and cannot be told apart from a dynamic one by mask presence. We
        therefore take the explicit ``static_names`` list (``vars["STATIC"]``, plumbed
        in from ``train_common``) and route those columns to ``static_idx``; their
        mask partners are dropped (mTAND does not model static missingness). Columns
        with no mask partner that are not in ``static_names`` are also treated as
        static/extra, preserving the previous behaviour.
        """
        if feature_names is None:
            raise ValueError(
                "MTANDNet requires `feature_names` (the data column names) to pair value/mask "
                "columns. It is provided by icu_benchmarks.models.train.train_common."
            )
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match the number of input "
                f"features ({num_features}). Did the GROUP column leak into feature_names?"
            )
        static_set = set(static_names or [])
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        value_idx, mask_idx, static_idx = [], [], []
        for i, name in enumerate(feature_names):
            if name.startswith(MISSING_PREFIX):
                continue  # consumed as the partner of its value column
            if name in static_set:
                static_idx.append(i)  # static value -> classifier branch; mask partner dropped
                continue
            partner = MISSING_PREFIX + name
            if partner in name_to_idx:
                value_idx.append(i)
                mask_idx.append(name_to_idx[partner])
            else:
                static_idx.append(i)
        if not value_idx:
            raise ValueError("MTANDNet found no value/mask column pairs; check the preprocessing pipeline.")
        return value_idx, mask_idx, static_idx

    @staticmethod
    def _log_normal_pdf(x, mean, logvar, mask):
        const = math.log(2.0 * math.pi)
        var = torch.exp(logvar) if torch.is_tensor(logvar) else math.exp(logvar)
        return -0.5 * (const + logvar + (x - mean) ** 2.0 / var) * mask

    @staticmethod
    def _normal_kl(mu1, lv1, mu2, lv2):
        v1, v2 = torch.exp(lv1), torch.exp(lv2)
        return lv2 / 2.0 - lv1 / 2.0 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5

    def forward(self, x):
        # x: (B, T, F). Split into time-series values, observation masks, statics.
        batch_size, seq_len, _ = x.shape
        value = x.index_select(-1, self.value_idx)  # (B, T, D)
        missing = x.index_select(-1, self.mask_idx)  # (B, T, D) 1 == originally missing
        mask = 1.0 - missing  # observed mask, 1 == observed
        mtan_in = torch.cat([value, mask], dim=-1)  # (B, T, 2D)

        # Encoder posterior over the latent state at the reference points.
        out = self.enc(mtan_in)  # (B, R, 2*latent)
        qz0_mean = out[:, :, : self.latent_dim]
        qz0_logvar = out[:, :, self.latent_dim:]

        # Importance-weighted latent samples (k_iwae) for the reconstruction ELBO.
        k = self.k_iwae
        eps = torch.randn(k, qz0_mean.size(0), qz0_mean.size(1), qz0_mean.size(2), device=x.device)
        z0 = eps * torch.exp(0.5 * qz0_logvar) + qz0_mean  # (k, B, R, latent)
        z0_flat = z0.view(-1, z0.size(2), z0.size(3))  # (k*B, R, latent)

        obs_tp = self.enc._obs_tp(x)  # (B, T) in [0, 1]
        obs_tp_rep = obs_tp.unsqueeze(0).repeat(k, 1, 1).view(-1, seq_len)  # (k*B, T)
        pred_x = self.dec(z0_flat, obs_tp_rep).view(k, batch_size, seq_len, -1)  # (k, B, T, D)

        aux_loss = self._reconstruction_elbo(value, mask, pred_x, qz0_mean, qz0_logvar, self._kl_coef()) / self.alpha

        # Classification head from a single latent sample (k_iwae=1 -> exact official path).
        static = x.index_select(-1, self.static_idx)[:, 0, :] if self.static_idx.numel() > 0 else None
        logits = self.clf(z0[0], static)  # (B, num_classes)

        # Broadcast across timesteps; step_fn scores only the labeled (last) step and adds aux_loss.
        pred = logits.unsqueeze(1).expand(batch_size, seq_len, logits.size(-1))
        return pred, aux_loss

    def _kl_coef(self):
        """KL annealing coefficient, per the official training loop / paper.

        ``kl_coef = 0`` for the first ``kl_wait`` epochs, then ``1 - kl_rate ** (epoch - kl_wait)``
        (the paper reports KL annealing with rate 0.99 improved classification). With
        ``kl_anneal=False`` it is fixed to 1 (the released command default). ``self.current_epoch``
        is provided by the Lightning trainer; falls back to a fully-annealed coef when unavailable.
        """
        if not self.kl_anneal:
            return 1.0
        try:
            epoch = self.current_epoch  # provided by the Lightning trainer
        except RuntimeError:
            return 1.0  # no trainer attached (e.g. unit test) -> fully-annealed coef
        if epoch < self.kl_wait:
            return 0.0
        return 1.0 - self.kl_rate ** (epoch - self.kl_wait)

    def _reconstruction_elbo(self, value, mask, pred_x, qz0_mean, qz0_logvar, kl_coef=1.0):
        """Negative IWAE bound: ``-(logsumexp_k(logpx - kl_coef * KL).mean - log k)``.

        Official formula (``tan_classification.py`` + ``utils.compute_losses``)
        """
        noise_logvar = 2.0 * math.log(self.noise_std)
        logpx = self._log_normal_pdf(value, pred_x, noise_logvar, mask).sum(-1).sum(-1)  # (k, B)
        analytic_kl = self._normal_kl(
            qz0_mean, qz0_logvar, torch.zeros_like(qz0_mean), torch.zeros_like(qz0_logvar)
        ).sum(-1).sum(-1)  # (B,)
        if self.norm:
            obs_count = mask.sum(-1).sum(-1).clamp(min=1.0)  # (B,)
            logpx = logpx / obs_count  # (k, B) / (B,) -> (k, B)
            analytic_kl = analytic_kl / obs_count  # (B,)
        logp = torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - math.log(self.k_iwae)
        return -logp
