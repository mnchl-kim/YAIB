"""SeFT (Set Functions for Time Series).

Reference:
    Horn, Moor, Bock, Rieck, Borgwardt, "Set Functions for Time Series",
    ICML 2020.
    Paper: https://proceedings.mlr.press/v119/horn20a/horn20a.pdf
    Preprint: https://arxiv.org/abs/1909.12064
    Official code: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
                   (seft/models/deep_set_attention.py)

What SeFT does
--------------
SeFT does *not* tensorize a time series into a regular ``B x T x F`` grid.
Instead it treats a series as an unordered **set** of observations, each an
``(t, value, modality)`` triplet, and learns a permutation-invariant set
function

    f(S) = g( aggregate_j { a_j * h(s_j) } )

where ``h`` (``phi``) embeds each observation, ``a_j`` are attention weights
produced by a learned set-attention head, the aggregation is a sum, and ``g``
(``rho``) maps the pooled representation to the prediction.  This file
implements the attention variant (``SeFT-Attn``, ``DeepSetAttentionModel`` in
the official repo) for the non-sequence (whole-series) prediction case.

Integration notes for YAIB
--------------------------
The YAIB preprocessing pipeline (see ``icu_benchmarks/data/preprocessor.py``)
forward/zero-fills every value column ``<f>`` and adds a paired binary column
``MissingIndicator_<f>`` (1 == originally missing).  The dense tensor that
reaches the model therefore lets us *reconstruct the original observation set*
without imputation: a value ``x[b, t, f]`` is a genuine observation iff its
mask says it was observed (``MissingIndicator_<f> == 0``).  We rebuild the set
on the fly from the regular grid and feed only the real observations to SeFT,
so the model never sees the forward-filled values.

We recover the (value, mask) pairs by name from ``feature_names`` (layout is
dataset/task agnostic).  ``dataset_vars`` tells us which columns are dynamic
(time-series modalities) and which are static (age/sex/height/weight); static
features are handled as demographics, encoded once and prepended as a single
extra set element, exactly as in the official implementation.

Observation times are the integer hour index on the regular 1-hour grid used
by YAIB-cohorts.  Padded timesteps (all-zero rows appended by the loader to
equalise sequence length) are detected and excluded from the set.

Prediction layout
------------------
This is the whole-series (non-sequence) SeFT: it aggregates the full set into a
single prediction.  To match the framework's ``(B, T, num_classes)`` output
interface (the loss masks to the labelled timesteps), the single prediction is
broadcast across time.  For ``mortality24`` the label sits only on the last
timestep, so this is exactly the intended whole-series classification.  Note:
genuine per-hour online tasks (aki/sepsis) would instead require the cumulative
variant from the paper to avoid using future observations; that is out of scope
for this baseline.
"""

import logging

import gin
import numpy as np
import torch
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

MISSING_PREFIX = "MissingIndicator_"


class PositionalEncoding(nn.Module):
    """Sinusoidal time encoding (official ``PositionalEncoding``).

    Encodes a scalar time ``t`` with ``n_dim`` features using
    ``n_dim // 2`` log-spaced timescales:
    ``[sin(t / s_k), cos(t / s_k)]_k`` with
    ``s_k = max_timescale ** (k / (n_dim//2 - 1))``.
    """

    def __init__(self, max_timescale: float, n_dim: int):
        super().__init__()
        if n_dim % 2 != 0:
            raise ValueError(f"n_positional_dims must be even, got {n_dim}.")
        num_timescales = n_dim // 2
        timescales = max_timescale ** np.linspace(0, 1, num_timescales)
        self.register_buffer("timescales", torch.tensor(timescales, dtype=torch.float32))

    def forward(self, times):
        # times: (..., ) -> scaled: (..., num_timescales)
        scaled = times.unsqueeze(-1) / self.timescales
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


def _build_mlp(in_dim: int, n_layers: int, width: int, dropout: float, out_dim: int):
    """ReLU MLP: ``n_layers`` hidden layers of ``width`` then a final ``out_dim`` layer.

    Mirrors the official ``build_dense_dropout_model`` followed by a final Dense.
    All layers (including the last) use ReLU, matching the SeFT encoders.
    """
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(d, width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, out_dim))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SetAttentionLayer(nn.Module):
    """Learned set-attention head pool (official ``SetAttentionLayer``).

    Per observation ``j`` and head ``i`` the (pre-)attention is a dot product
    between a single learned query ``W_q[i]`` (initialised to zero, so attention
    starts uniform == mean pooling) and a key computed from the concatenation of
    the observation and the set's psi-mean context:

        e_j        = psi(s_j)
        context    = rho_attn( mean_j e_j )            # broadcast to every j
        key_j      = [s_j, context] @ W_k              # split into heads
        preattn_ji = <key_ji, W_q[i]> / sqrt(d)

    followed by a masked softmax over the set (per head).
    """

    def __init__(self, input_dim, n_layers, width, latent_width, dot_prod_dim, n_heads, attn_dropout):
        super().__init__()
        self.dot_prod_dim = dot_prod_dim
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.psi = _build_mlp(input_dim, n_layers, width, 0.0, latent_width)
        self.rho = nn.Sequential(nn.Linear(latent_width, latent_width), nn.ReLU())
        self.W_k = nn.Parameter(torch.empty(input_dim + latent_width, dot_prod_dim * n_heads))
        nn.init.kaiming_uniform_(self.W_k, nonlinearity="relu")
        # Zero init -> uniform attention at start (mean aggregation), as in the paper.
        self.W_q = nn.Parameter(torch.zeros(n_heads, dot_prod_dim))

    def forward(self, inputs, obs_mask):
        # inputs: (B, N, input_dim); obs_mask: (B, N) 1 == real observation.
        m = obs_mask.unsqueeze(-1)  # (B, N, 1)
        encoded = self.psi(inputs)  # (B, N, latent)
        denom = obs_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        agg = (encoded * m).sum(dim=1) / denom  # (B, latent) masked mean
        agg = self.rho(agg)  # (B, latent)
        combined = torch.cat([inputs, agg.unsqueeze(1).expand(-1, inputs.size(1), -1)], dim=-1)
        keys = combined @ self.W_k  # (B, N, dot_prod_dim * n_heads)
        keys = keys.view(keys.size(0), keys.size(1), self.n_heads, self.dot_prod_dim)
        # preattn: (B, N, n_heads)
        preattn = torch.einsum("bnhd,hd->bnh", keys, self.W_q) / np.sqrt(self.dot_prod_dim)

        neg_inf = torch.finfo(preattn.dtype).min
        invalid = obs_mask == 0  # (B, N)
        if self.training and self.attn_dropout > 0:
            drop = torch.rand_like(obs_mask) < self.attn_dropout
            invalid = invalid | drop
        preattn = preattn.masked_fill(invalid.unsqueeze(-1), neg_inf)
        attn = torch.softmax(preattn, dim=1)  # (B, N, n_heads) masked softmax over set
        return attn, m


@gin.configurable
class SeFTNet(DLPredictionWrapper):
    """SeFT-Attn model for irregularly sampled clinical time series.

    Rebuilds the observation set from the YAIB (value, mask) grid, encodes each
    observation with ``phi``, pools with multi-head set attention, and maps the
    pooled vector to a single prediction via ``rho`` (broadcast across time to
    match the framework output interface).
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        num_classes,
        *args,
        feature_names=None,
        dataset_vars=None,
        # phi (observation encoder)
        n_phi_layers: int = 3,
        phi_width: int = 32,
        phi_dropout: float = 0.0,
        latent_width: int = 128,
        # set-attention pool
        n_psi_layers: int = 2,
        psi_width: int = 64,
        psi_latent_width: int = 128,
        dot_prod_dim: int = 128,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        # rho (output decoder)
        n_rho_layers: int = 3,
        rho_width: int = 32,
        rho_dropout: float = 0.0,
        # time encoding
        max_timescale: float = 100.0,
        n_positional_dims: int = 16,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            num_classes=num_classes,
            feature_names=feature_names,
            **kwargs,
        )
        self.num_classes = num_classes
        self.n_heads = n_heads

        num_features = input_size[2]
        seq_len = input_size[1]
        value_idx, mask_idx, static_value_idx = self._build_feature_index(feature_names, num_features, dataset_vars)
        self.register_buffer("value_idx", torch.tensor(value_idx, dtype=torch.long))
        self.register_buffer("mask_idx", torch.tensor(mask_idx, dtype=torch.long))
        self.register_buffer("static_idx", torch.tensor(static_value_idx, dtype=torch.long))

        n_modalities = len(value_idx)  # one modality per dynamic feature
        n_static = len(static_value_idx)
        logging.info(
            f"SeFTNet: {num_features} input features -> {n_modalities} dynamic modalities, "
            f"{n_static} static (demographic) features; seq_len={seq_len}."
        )

        self.positional_encoding = PositionalEncoding(max_timescale, n_positional_dims)
        # Per-observation input: [time_enc (P), value (1), modality one-hot (n_modalities)].
        obs_input_dim = n_positional_dims + 1 + n_modalities

        # Precompute the dense candidate grid layout (t outer, modality inner),
        # matching the row-major flatten of (B, T, D) used in forward.
        times = torch.arange(seq_len, dtype=torch.float32).repeat_interleave(n_modalities)  # (T*D,)
        self.register_buffer("cand_time_enc", self.positional_encoding(times))  # (T*D, P)
        modality_onehot = torch.eye(n_modalities).repeat(seq_len, 1)  # (T*D, D)
        self.register_buffer("modality_onehot", modality_onehot)

        # Demographics encoder: maps static features to one extra set element of obs_input_dim.
        self.has_demo = n_static > 0
        if self.has_demo:
            self.demo_encoder = nn.Sequential(
                nn.Linear(n_static, phi_width), nn.ReLU(), nn.Linear(phi_width, obs_input_dim)
            )

        self.phi = _build_mlp(obs_input_dim, n_phi_layers, phi_width, phi_dropout, latent_width)
        self.attention = SetAttentionLayer(
            obs_input_dim, n_psi_layers, psi_width, psi_latent_width, dot_prod_dim, n_heads, attn_dropout
        )

        # rho (g): `n_rho_layers` hidden ReLU layers then the output layer, matching the
        # official `build_dense_dropout_model(n_rho_layers, rho_width) + Dense(output_dims)`.
        # The output layer is `self.logit` (named as the wrapper expects); YAIB uses
        # `num_classes` logits + CrossEntropy instead of the paper's sigmoid/softmax head.
        rho_in = latent_width * n_heads
        rho_layers = []
        d = rho_in
        for _ in range(n_rho_layers):
            rho_layers.append(nn.Linear(d, rho_width))
            rho_layers.append(nn.ReLU())
            if rho_dropout > 0:
                rho_layers.append(nn.Dropout(rho_dropout))
            d = rho_width
        self.rho_body = nn.Sequential(*rho_layers)
        self.logit = nn.Linear(d, num_classes)

    @staticmethod
    def _build_feature_index(feature_names, num_features, dataset_vars):
        """Return index lists into the input tensor: dynamic value cols, their
        matching ``MissingIndicator_`` mask cols, and static value cols.

        Dynamic vs static is decided from ``dataset_vars`` (``vars["DYNAMIC"]`` /
        ``vars["STATIC"]``). Every value column is expected to have a mask
        partner produced by the preprocessor.
        """
        if feature_names is None or dataset_vars is None:
            raise ValueError(
                "SeFTNet requires `feature_names` and `dataset_vars`; they are provided by "
                "icu_benchmarks.models.train.train_common."
            )
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) != number of input features "
                f"({num_features}). Did the GROUP column leak into feature_names?"
            )
        static_set = set(dataset_vars.get("STATIC", []) or [])
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        value_idx, mask_idx, static_value_idx = [], [], []
        for i, name in enumerate(feature_names):
            if name.startswith(MISSING_PREFIX):
                continue  # consumed as the partner of its value column
            partner = MISSING_PREFIX + name
            if partner not in name_to_idx:
                # A value column without a mask; treat as static if known, else as a
                # maskless static feature (always observed).
                static_value_idx.append(i)
                continue
            if name in static_set:
                static_value_idx.append(i)
            else:
                value_idx.append(i)
                mask_idx.append(name_to_idx[partner])
        if not value_idx:
            raise ValueError("SeFTNet found no dynamic value/mask column pairs; check preprocessing.")
        return value_idx, mask_idx, static_value_idx

    def forward(self, x):
        # x: (B, T, F) regular grid with forward-filled values + MissingIndicator masks.
        batch_size, seq_len, _ = x.shape
        n_mod = self.value_idx.numel()

        values = x.index_select(-1, self.value_idx)  # (B, T, D) forward-filled values
        missing = x.index_select(-1, self.mask_idx)  # (B, T, D) 1 == originally missing
        observed = 1.0 - missing  # 1 == real observation

        # Exclude padded timesteps (all-zero rows appended by the loader).
        timestep_valid = (x.abs().sum(dim=-1) > 0).float().unsqueeze(-1)  # (B, T, 1)
        obs_mask = (observed * timestep_valid).reshape(batch_size, seq_len * n_mod)  # (B, T*D)

        # Build per-observation inputs over the dense candidate grid.
        time_enc = self.cand_time_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T*D, P)
        modality = self.modality_onehot.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T*D, D)
        value_flat = values.reshape(batch_size, seq_len * n_mod, 1)  # (B, T*D, 1)
        obs_inputs = torch.cat([time_enc, value_flat, modality], dim=-1)  # (B, T*D, P+1+D)

        # Prepend the demographics element (always a valid set member).
        if self.has_demo:
            demo = x[:, 0, :].index_select(-1, self.static_idx)  # (B, S) constant across time
            demo_enc = self.demo_encoder(demo).unsqueeze(1)  # (B, 1, P+1+D)
            obs_inputs = torch.cat([demo_enc, obs_inputs], dim=1)
            demo_mask = obs_mask.new_ones(batch_size, 1)
            obs_mask = torch.cat([demo_mask, obs_mask], dim=1)

        attn, m = self.attention(obs_inputs, obs_mask)  # attn: (B, N, n_heads), m: (B, N, 1)
        encoded = self.phi(obs_inputs)  # (B, N, latent)

        # Per-head attention-weighted sum aggregation, then concatenate heads.
        weighted = encoded.unsqueeze(-1) * attn.unsqueeze(2) * m.unsqueeze(-1)  # (B, N, latent, n_heads)
        pooled = weighted.sum(dim=1)  # (B, latent, n_heads)
        pooled = pooled.reshape(batch_size, -1)  # (B, latent * n_heads)

        rep = self.rho_body(pooled)
        logits = self.logit(rep)  # (B, num_classes) whole-series prediction

        # Broadcast across time to match the (B, T, num_classes) output interface.
        return logits.unsqueeze(1).expand(-1, seq_len, -1)
