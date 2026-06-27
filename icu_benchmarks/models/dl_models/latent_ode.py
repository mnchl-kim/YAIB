"""Latent ODE for irregularly-sampled clinical time series.

Reference:
    Rubanova, Chen, Duvenaud, "Latent ODEs for Irregularly-Sampled Time Series",
    NeurIPS 2019.
    Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf
    Preprint: https://arxiv.org/abs/1907.03907
    Official code: https://github.com/YuliaRubanova/latent_ode

This is a faithful port of the official PhysioNet *classification* recipe
(``--dataset physionet``), which is the configuration closest to YAIB's
mortality/AKI/sepsis tasks (one binary label per sequence). Concretely we mirror
the official ``lib/`` components:

* ``Encoder_z0_ODE_RNN`` + the masked Gaussian ``GRU_unit`` run **backward** to
  infer ``q(z0) = N(mean, std)`` (encoder ODE solved with Euler, as in the
  original ``z0_diffeq_solver``).
* ``z0`` is sampled with the reparameterisation trick (``n_traj_samples`` draws,
  IWAE objective).
* a generative ODE (Dopri5) decodes the latent trajectory; a linear ``Decoder``
  reconstructs the observations.
* classification is done from ``z0`` directly (``classif_per_tp = False``,
  ``self.classifier(first_point_enc)``) with the 3-layer MLP from
  ``create_classifier``.
* the loss is the official ELBO ``-logsumexp(rec_likelihood - kl_coef*KL)`` plus
  the classification CE; the official combines them as ``ELBO + 100*CE``.

YAIB integration
-----------------
``DLPredictionWrapper.step_fn`` computes the (class-weighted, 2-class softmax) CE
itself over the labelled timesteps and adds the second element of the forward
tuple as ``aux_loss``. So:

* ``forward`` returns ``(out, aux_loss)`` where ``out`` is the z0 classification
  logits broadcast over the time axis -> ``(B, T, num_classes)``. The framework's
  mask selects the single labelled (last) timestep, i.e. it scores the z0
  prediction. This is the YAIB-consistent CE (matching GRU/GRU-D), in place of
  the official unweighted ``BCEWithLogitsLoss``.
* ``aux_loss = reconstr_coef * ELBO``. With ``reconstr_coef = 0.01`` the total
  ``CE + 0.01*ELBO`` is proportional to the official ``ELBO + 100*CE`` (the
  global scale is absorbed by the learning rate), reproducing the official
  reconstruction/KL-vs-classification balance.

Like GRU-D we recover (value, mask) pairs from ``feature_names`` via the
``MissingIndicator_<col>`` convention emitted by the YAIB preprocessor. The YAIB
preprocessor applies a ``MissingIndicator`` to *both* dynamic and static columns,
so the naming convention alone cannot tell static (``age``/``sex``/``height``/
``weight``) value columns apart from dynamic ones. We therefore take the explicit
``static_names`` list (``vars["STATIC"]``, plumbed in from ``train_common``) and
route those columns to ``sta_idx`` so they stay explicitly separate from the
dynamic ones. Static columns are appended to the reconstructed ``data`` but keep
their real ``MissingIndicator_<col>`` mask (via ``sta_mask_idx``), so a static
value that was actually missing for a patient is still masked out; a column with
no mask partner falls back to always-observed. The regular 1-hour YAIB-cohorts
grid is normalised to ``[0, 1]`` for numerically stable ODE integration.

KL annealing follows the official schedule (``kl_coef = 0`` for the first
``kl_warmup_epochs`` epochs, then ``1 - 0.99**(epoch - kl_warmup_epochs)``),
driven by the Lightning ``current_epoch``.

The optional Poisson process likelihood (``use_poisson``, official ``--poisson``)
models the observation *times* as an inhomogeneous Poisson process whose rate
``lambda(t)`` is parameterised from the latent trajectory -- directly encoding the
idea that *when* a measurement is taken is informative. When enabled, the
generative latent doubles (``[reconstruction | rate]``) and the loss gains a
``- poisson_coef * Poisson-log-likelihood`` term. NOTE: the official paper reports
this term did **not** improve PhysioNet classification accuracy, so it is exposed
as a tunable on/off switch rather than always-on.

The only remaining deviation from the official setup is the loss head: YAIB's
class-weighted 2-class softmax CE (added by ``step_fn``) is used in place of the
official unweighted ``BCEWithLogitsLoss``, for consistency with how GRU/GRU-D are
scored in this benchmark.
"""

import logging

import gin
import torch
from torch import nn as nn
from torchdiffeq import odeint

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

MISSING_PREFIX = "MissingIndicator_"


def init_network_weights(net, std: float = 0.1):
    """Official ``utils.init_network_weights``: small-normal weights, zero bias."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            nn.init.constant_(m.bias, val=0.0)


def create_net(n_inputs, n_outputs, n_layers: int = 1, n_units: int = 100, nonlinear=nn.Tanh):
    """Official ``utils.create_net`` (verbatim structure)."""
    layers = [nn.Linear(n_inputs, n_units)]
    for _ in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))
    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def split_last_dim(data):
    """Official ``utils.split_last_dim``: split the last axis into two halves."""
    last_dim = data.size(-1) // 2
    return data[..., :last_dim], data[..., last_dim:]


def create_classifier(z0_dim, n_labels):
    """Official ``base_models.create_classifier`` (3-layer MLP, 300 units)."""
    return nn.Sequential(
        nn.Linear(z0_dim, 300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Linear(300, n_labels),
    )


class ODEFunc(nn.Module):
    """Neural ODE gradient field ``dy/dt = net(y)`` (official ``ode_func.ODEFunc``)."""

    def __init__(self, ode_func_net):
        super().__init__()
        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards: bool = False):
        grad = self.gradient_net(y)
        return -grad if backwards else grad


class ODEFuncPoisson(ODEFunc):
    """Augmented ODE-func modelling observation times as a Poisson process.

    Faithful port of the official ``ode_func.ODEFunc_w_Poisson``. The augmented
    state is ``[y_latent_lam (2*latents) | int_lambda (data_dim)]`` where
    ``y_latent_lam = [y (latents, reconstruction) | y_lambda (latents, rate)]``.
    The rate ``lambda(t) = exp(lambda_net(y_lambda))`` and ``int_lambda`` integrates
    it; ``const_for_lambda`` keeps the integral numerically stable.
    """

    def __init__(self, data_dim, latent_dim, ode_func_net, lambda_net):
        super().__init__(ode_func_net)
        self.data_dim = data_dim
        self.latent_dim = latent_dim  # = 2 * latents (state evolved by the ODE)
        self.lambda_net = lambda_net
        init_network_weights(self.lambda_net)
        self.register_buffer("const_for_lambda", torch.tensor(100.0))

    def extract_poisson_rate(self, augmented, final_result: bool = True):
        # augmented last dim == latent_dim + data_dim
        latent_lam_dim = self.latent_dim // 2  # = latents
        int_lambda = augmented[..., -self.data_dim:]
        y_latent_lam = augmented[..., :-self.data_dim]
        log_lambdas = self.lambda_net(y_latent_lam[..., -latent_lam_dim:])  # rate latents -> data_dim
        y = y_latent_lam[..., :latent_lam_dim]                              # reconstruction latents
        if final_result:
            int_lambda = int_lambda * self.const_for_lambda
        return y, log_lambdas, int_lambda, y_latent_lam

    def forward(self, t_local, augmented, backwards: bool = False):
        _, log_lam, _, y_latent_lam = self.extract_poisson_rate(augmented, final_result=False)
        dydt_dldt = self.gradient_net(y_latent_lam)              # d[y, y_lambda]/dt  (2*latents)
        log_lam = log_lam - torch.log(self.const_for_lambda)
        grad = torch.cat((dydt_dldt, torch.exp(log_lam)), -1)   # append d(int_lambda)/dt = lambda
        return -grad if backwards else grad


class DiffeqSolver(nn.Module):
    """Integrate an :class:`ODEFunc` with ``torchdiffeq.odeint`` (official ``DiffeqSolver``).

    Works for any leading batch shape: ``first_point`` ``(..., d)`` and
    ``time_steps`` ``(T,)`` -> trajectory ``(..., T, d)``. The official encoder
    solver uses ``euler`` while the generative solver uses ``dopri5``.
    """

    def __init__(self, ode_func: ODEFunc, method: str = "dopri5", ode_steps: int = 2,
                 rtol: float = 1e-3, atol: float = 1e-4):
        super().__init__()
        self.ode_func = ode_func
        self.method = method
        self.ode_steps = max(1, ode_steps)
        self.rtol = rtol
        self.atol = atol

    def _options(self, time_steps):
        if self.method in ("euler", "rk4", "midpoint"):
            span = (time_steps[-1] - time_steps[0]).abs().item()
            n_intervals = max(1, time_steps.numel() - 1)
            return {"step_size": span / n_intervals / self.ode_steps}
        return None

    def forward(self, first_point, time_steps):
        pred = odeint(self.ode_func, first_point, time_steps, rtol=self.rtol,
                      atol=self.atol, method=self.method, options=self._options(time_steps))
        # odeint returns (T, *batch, d); move time next to the feature axis.
        return pred.movedim(0, -2)


class GRUUnit(nn.Module):
    """Masked Gaussian GRU cell propagating (mean, std) (official ``encoder_decoder.GRU_unit``)."""

    def __init__(self, hidden_dim: int, input_dim: int, n_units: int = 100):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, hidden_dim), nn.Sigmoid())
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, hidden_dim), nn.Sigmoid())
        self.new_state_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, hidden_dim * 2))
        init_network_weights(self.update_gate)
        init_network_weights(self.reset_gate)
        init_network_weights(self.new_state_net)

    def forward(self, y_mean, y_std, x, masked_update: bool = True):
        y_concat = torch.cat([y_mean, y_std, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        if masked_update:
            # x is [data; mask]; update the hidden state only where >=1 feature is observed.
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std

        return new_y, new_y_std.abs()


class EncoderZ0ODERNN(nn.Module):
    """Backward ODE-RNN recognition net inferring ``q(z0)`` (official ``Encoder_z0_ODE_RNN``).

    Dimension naming (kept consistent across the model):
        rec_dim   - the recognition ODE-RNN hidden / ODE-state dimension.
        z0_dim    - the dimension of the inferred initial latent ``q(z0)`` (the
                    generative ``latent_dim`` in :class:`LatentODENet`).
        input_dim - the per-step encoder input width, i.e. ``[values; masks]``.
    """

    def __init__(self, rec_dim: int, input_dim: int, z0_dim: int,
                 solver: DiffeqSolver, n_gru_units: int = 100):
        super().__init__()
        self.rec_dim = rec_dim  # recognition ODE-RNN hidden / ODE-state dimension
        self.z0_dim = z0_dim    # dimension of the inferred q(z0)
        self.gru_update = GRUUnit(rec_dim, input_dim, n_units=n_gru_units)
        self.solver = solver
        self.transform_z0 = nn.Sequential(
            nn.Linear(rec_dim * 2, 100), nn.Tanh(), nn.Linear(100, z0_dim * 2))
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps):
        """data: (B, T, input_dim) with input = [values, masks]; processed in reverse time."""
        batch_size, seq_len, _ = data.shape
        prev_y = data.new_zeros(1, batch_size, self.rec_dim)
        prev_std = data.new_zeros(1, batch_size, self.rec_dim)
        # Normalised regular grid -> constant interval; the ODE is autonomous so
        # only the span matters. Span between adjacent grid points:
        dt = (time_steps[1] - time_steps[0]).abs() if seq_len > 1 else time_steps.new_tensor(1.0)
        interval = torch.stack([time_steps.new_zeros(()), dt])

        for step, i in enumerate(reversed(range(seq_len))):
            if step > 0:
                ode_sol = self.solver(prev_y, interval)  # (1, B, 2, rec_dim)
                yi_ode = ode_sol[:, :, -1, :]
            else:
                yi_ode = prev_y
            xi = data[:, i, :].unsqueeze(0)  # (1, B, input_dim)
            prev_y, prev_std = self.gru_update(yi_ode, prev_std, xi)

        mean_z0, std_z0 = split_last_dim(self.transform_z0(torch.cat((prev_y, prev_std), -1)))
        return mean_z0, std_z0.abs()  # each (1, B, z0_dim)


@gin.configurable
class LatentODENet(DLPredictionWrapper):
    """Latent ODE classifier for irregularly-sampled clinical time series."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        latent_dim,
        rec_dim,
        units,
        num_classes,
        *args,
        feature_names=None,
        static_names=None,
        gru_units: int = 100,
        gen_layers: int = 1,
        rec_layers: int = 1,
        n_traj_samples: int = 3,
        obsrv_std: float = 0.01,
        kl_warmup_epochs: int = 10,
        reconstr_coef: float = 0.01,
        enc_ode_steps: int = 2,
        use_poisson: bool = True,
        poisson_coef: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            latent_dim=latent_dim,
            rec_dim=rec_dim,
            units=units,
            num_classes=num_classes,
            feature_names=feature_names,
            static_names=static_names,
            gru_units=gru_units,
            gen_layers=gen_layers,
            rec_layers=rec_layers,
            n_traj_samples=n_traj_samples,
            obsrv_std=obsrv_std,
            kl_warmup_epochs=kl_warmup_epochs,
            reconstr_coef=reconstr_coef,
            enc_ode_steps=enc_ode_steps,
            use_poisson=use_poisson,
            poisson_coef=poisson_coef,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.n_traj_samples = n_traj_samples
        self.kl_warmup_epochs = kl_warmup_epochs
        self.reconstr_coef = reconstr_coef
        self.use_poisson = bool(use_poisson)
        self.poisson_coef = poisson_coef
        self.register_buffer("obsrv_std", torch.tensor(float(obsrv_std)))

        num_features = input_size[2]
        dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx = self._build_feature_index(
            feature_names, num_features, static_names
        )
        self.register_buffer("dyn_idx", torch.tensor(dyn_idx, dtype=torch.long))
        self.register_buffer("dyn_mask_idx", torch.tensor(dyn_mask_idx, dtype=torch.long))
        self.register_buffer("sta_idx", torch.tensor(sta_idx, dtype=torch.long))
        self.register_buffer("sta_mask_idx", torch.tensor(sta_mask_idx, dtype=torch.long))

        # Reconstructed data = dynamic value/mask pairs + always-observed static extras.
        data_dim = len(dyn_idx) + len(sta_idx)
        self.data_dim = data_dim
        enc_input_dim = data_dim * 2  # [data; mask], official enc_input_dim = input_dim * 2
        logging.info(
            f"LatentODENet: {num_features} features -> {len(dyn_idx)} dynamic value/mask pairs + "
            f"{len(sta_idx)} static value/mask pairs; data_dim={data_dim}, latent_dim={latent_dim}, "
            f"rec_dim={rec_dim}, n_traj_samples={n_traj_samples}."
        )

        # With the Poisson process the z0 / generative latent doubles: [reconstruction | rate].
        z0_eff = 2 * latent_dim if self.use_poisson else latent_dim
        self.z0_eff = z0_eff

        # Recognition net: its own ODE solved with Euler (as in the official z0_diffeq_solver).
        enc_ode = ODEFunc(create_net(rec_dim, rec_dim, n_layers=rec_layers, n_units=units))
        enc_solver = DiffeqSolver(enc_ode, method="euler", ode_steps=enc_ode_steps)
        self.encoder = EncoderZ0ODERNN(
            rec_dim=rec_dim, input_dim=enc_input_dim, z0_dim=z0_eff,
            solver=enc_solver, n_gru_units=gru_units,
        )

        # Generative net: Dopri5 (official default), plus a linear reconstruction decoder.
        if self.use_poisson:
            gen_ode = ODEFuncPoisson(
                data_dim,
                2 * latent_dim,
                create_net(2 * latent_dim, 2 * latent_dim, n_layers=gen_layers, n_units=units),
                lambda_net=create_net(latent_dim, data_dim, n_layers=1, n_units=units),
            )
        else:
            gen_ode = ODEFunc(create_net(latent_dim, latent_dim, n_layers=gen_layers, n_units=units))
        self.gen_ode = gen_ode
        self.gen_solver = DiffeqSolver(gen_ode, method="dopri5", rtol=1e-3, atol=1e-4)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, data_dim))  # decodes reconstruction latents
        init_network_weights(self.decoder)

        # Classifier from z0 (official create_classifier); final layer named `logit` for set_metrics.
        self.classifier = create_classifier(z0_eff, num_classes)
        init_network_weights(self.classifier)
        self.logit = self.classifier[-1]

    @staticmethod
    def _build_feature_index(feature_names, num_features, static_names=None):
        """Split columns into dynamic/static value/mask pairs.

        Returns:
            dyn_idx, dyn_mask_idx: feature indices of the dynamic value columns and their mask partners.
            sta_idx, sta_mask_idx: feature indices of the static value columns and their mask partners.
        """
        if feature_names is None:
            raise ValueError(
                "LatentODENet requires `feature_names` to pair value/mask columns. "
                "It is provided by icu_benchmarks.models.train.train_common."
            )
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match the number of input "
                f"features ({num_features}). Did the GROUP column leak into feature_names?"
            )
        static_set = set(static_names or [])
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx = [], [], [], []
        for i, name in enumerate(feature_names):
            if name.startswith(MISSING_PREFIX):
                continue
            partner = name_to_idx.get(MISSING_PREFIX + name, -1)
            if partner >= 0:
                if name in static_set:
                    sta_idx.append(i)
                    sta_mask_idx.append(partner)
                else:
                    dyn_idx.append(i)
                    dyn_mask_idx.append(partner)
            else:
                raise ValueError(f"Column {name} has no MissingIndicator partner; check the preprocessing pipeline.")
        if not dyn_idx:
            raise ValueError("LatentODENet found no value/mask column pairs; check the preprocessing pipeline.")
        return dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx

    def _split_input(self, x):
        """Return (data, mask) each (B, T, data_dim): forward-filled values+static, observed mask."""
        dynamic = x.index_select(-1, self.dyn_idx)            # (B, T, D)
        dynamic_mask = 1.0 - x.index_select(-1, self.dyn_mask_idx)    # 1 == observed
        if self.sta_idx.numel() > 0:
            static = x.index_select(-1, self.sta_idx)                  # static values
            static_mask = 1.0 - x.index_select(-1, self.sta_mask_idx)  # each static column's MI mask
            data = torch.cat([dynamic, static], dim=-1)
            mask = torch.cat([dynamic_mask, static_mask], dim=-1)
        else:
            data, mask = dynamic, dynamic_mask
        return data, mask

    def _current_kl_coef(self):
        """Official KL annealing (run_models.py): 0 during warmup, then 1 - 0.99**(epoch - warmup).

        ``current_epoch`` is driven by the Lightning trainer during fit/validate/test;
        it safely returns 0 when no trainer is attached (e.g. unit tests).
        """
        epoch = int(self.current_epoch)
        if epoch < self.kl_warmup_epochs:
            return 0.0
        return 1.0 - 0.99 ** (epoch - self.kl_warmup_epochs)

    def _gaussian_log_likelihood(self, pred_x, data, mask):
        """Vectorised masked Gaussian log-density matching the official semantics.

        Official: per (sample, traj, feature) it is the *mean* over observed
        timepoints of the pointwise ``N(data | mu, obsrv_std)`` log-density; then
        mean over features, then mean over trajectories -> one value per sample.

        Args:
            pred_x: (K, B, T, D) reconstruction.
            data: (B, T, D) forward-filled values.
            mask: (B, T, D) observed mask.
        Returns:
            (K,) reconstruction log-likelihood per trajectory sample.
        """
        data = data.unsqueeze(0)  # (1, B, T, D) -> broadcasts over K
        mask = mask.unsqueeze(0)
        std = self.obsrv_std
        pointwise = -0.5 * ((data - pred_x) / std) ** 2 - 0.5 * torch.log(2 * torch.pi * std ** 2)
        n_obs = mask.sum(dim=2)                                   # (K, B, D) observed count per feature
        per_feature = (pointwise * mask).sum(dim=2) / n_obs.clamp(min=1.0)  # mean over observed tp
        per_feature = torch.where(n_obs > 0, per_feature, torch.zeros_like(per_feature))
        per_traj = per_feature.mean(dim=-1)                      # mean over features -> (K, B)
        return per_traj.mean(dim=-1)                             # mean over trajectories -> (K,)

    def _poisson_log_likelihood(self, log_lambda_y, int_lambda_last, mask):
        """Inhomogeneous Poisson process log-likelihood of the observation times.

        Per feature: ``sum_{observed tp} log lambda(t) - integral lambda(t) dt`` (official
        ``compute_poisson_proc_likelihood``). Averaged over features / trajectories / samples.

        Args:
            log_lambda_y: (K, B, T, D) log-rate at every timepoint.
            int_lambda_last: (K, B, D) integrated rate at the final timepoint.
            mask: (B, T, D) observed mask.
        Returns:
            scalar Poisson log-likelihood.
        """
        sum_log_lam = (log_lambda_y * mask.unsqueeze(0)).sum(dim=2)  # (K, B, D) sum over observed tp
        per_feature = sum_log_lam - int_lambda_last                 # (K, B, D)
        return per_feature.mean()                                   # mean over samples/traj/features

    def forward(self, x):
        # x: (B, T, F). Recover values, observed mask and static extras.
        batch_size, seq_len, _ = x.shape
        data, mask = self._split_input(x)  # each (B, T, data_dim)

        # Normalised, evenly-spaced observation times on the regular YAIB grid.
        time_steps = torch.linspace(0.0, 1.0, seq_len, device=x.device, dtype=x.dtype)

        # --- Recognition: q(z0) = N(mean, std) ---
        enc_input = torch.cat([data, mask], dim=-1)              # (B, T, 2*data_dim)
        mean_z0, std_z0 = self.encoder(enc_input, time_steps)   # each (1, B, latent)

        K = self.n_traj_samples
        means = mean_z0.repeat(K, 1, 1)                         # (K, B, z0_eff)
        stds = std_z0.repeat(K, 1, 1)
        z0 = means + stds * torch.randn_like(stds)              # reparameterised samples

        # --- Generative ODE + reconstruction ---
        if self.use_poisson:
            # Augment latent with the (zero-initialised) integrated Poisson rate, integrate, split.
            aug = torch.cat([z0, z0.new_zeros(K, batch_size, self.data_dim)], dim=-1)
            sol = self.gen_solver(aug, time_steps)              # (K, B, T, 2*latent + data_dim)
            y, log_lambda_y, int_lambda, _ = self.gen_ode.extract_poisson_rate(sol)
            pred_x = self.decoder(y)                            # (K, B, T, data_dim)
        else:
            sol_y = self.gen_solver(z0, time_steps)             # (K, B, T, latent)
            pred_x = self.decoder(sol_y)                        # (K, B, T, data_dim)

        # --- ELBO (IWAE) as aux loss ---
        rec_ll = self._gaussian_log_likelihood(pred_x, data, mask)   # (K,)
        var = std_z0 ** 2
        kldiv = 0.5 * (mean_z0 ** 2 + var - torch.log(var + 1e-12) - 1.0)  # (1, B, z0_eff)
        kldiv = kldiv.mean(dim=(1, 2)).repeat(K)                     # (K,) KL(q||N(0,1))
        kl_coef = self._current_kl_coef()                            # official annealing schedule
        elbo = -torch.logsumexp(rec_ll - kl_coef * kldiv, dim=0)
        if torch.isnan(elbo):
            elbo = -torch.mean(rec_ll - kl_coef * kldiv, dim=0)
        if self.use_poisson:
            pois_ll = self._poisson_log_likelihood(log_lambda_y, int_lambda[:, :, -1, :], mask)
            aux_loss = self.reconstr_coef * (elbo - self.poisson_coef * pois_ll)
        else:
            aux_loss = self.reconstr_coef * elbo

        # --- Classification from z0 (classif_per_tp = False) ---
        if self.training:
            logits = self.classifier(z0).mean(dim=0)            # (B, num_classes), avg over samples
        else:
            logits = self.classifier(mean_z0).squeeze(0)        # deterministic (mean z0)
        out = logits.unsqueeze(1).expand(batch_size, seq_len, logits.size(-1))  # (B, T, num_classes)
        return out, aux_loss
