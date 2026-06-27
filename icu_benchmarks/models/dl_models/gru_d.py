"""GRU-D (Gated Recurrent Unit with trainable Decays).

Reference:
    Che et al., "Recurrent Neural Networks for Multivariate Time Series with
    Missing Values", Scientific Reports 8, 6085 (2018).
    Paper: https://www.nature.com/articles/s41598-018-24271-9
    Preprint: https://arxiv.org/abs/1606.01865
    Official/reference code: https://github.com/zhiyongc/GRU-D ,
                             https://github.com/PeterChe1990/GRU-D

Integration notes for YAIB
--------------------------
The YAIB preprocessing pipeline (see ``icu_benchmarks/data/preprocessor.py``)
produces, for every input feature column ``<f>``, a paired binary column
``MissingIndicator_<f>`` (1 == originally missing) and then forward-fills and
zero-fills the value column.  Hence the tensor that reaches the model already
contains, per feature:

    * the forward-filled (last-observation-carried-forward) value, which is
      exactly the ``x'`` term GRU-D needs for input decay, and
    * its missingness mask (``m = 1 - MissingIndicator``).

We recover the (value, mask) pairs by name from ``feature_names`` so the layout
is dataset/task agnostic.  The preprocessor masks both dynamic and static columns,
so ``static_names`` (``vars["STATIC"]``, plumbed from ``train_common``) is used to
tell them apart; both are still fed to the GRU-D cell as time series (the official
GRU-D makes no such distinction), so the split is bookkeeping only.  Time deltas
``delta`` are derived inside the model from the mask on the regular 1-hour grid used
by YAIB-cohorts.  The empirical mean GRU-D decays towards is 0 under YAIB's standard
scaling (see ``GRUDCell.__init__``).
"""

import logging

import gin
import torch
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

MISSING_PREFIX = "MissingIndicator_"


class GRUDCell(nn.Module):
    """A single GRU-D recurrent cell with trainable input and hidden decays.

    Args:
        input_dim: number of decayable features ``D`` (value/mask pairs).
        hidden_dim: hidden state size ``H``.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU gate weights; gate input is [x_hat (D), mask (D)] so the mask is fed
        # into every gate (Che et al.).
        self.x2h = nn.Linear(2 * input_dim, 3 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim)

        # Input decay (diagonal W_γx -> per-feature vector) and hidden decay (full W_γh).
        self.gamma_x_w = nn.Parameter(torch.zeros(input_dim))
        self.gamma_x_b = nn.Parameter(torch.zeros(input_dim))
        self.gamma_h_lin = nn.Linear(input_dim, hidden_dim)

        # Empirical mean to decay missing values towards. The YAIB preprocessor
        # standard-scales features on their observed values (StandardScaler,
        # NaN-ignoring fit), so observed means are 0 and the paper's x̃ == 0.
        # Verified on hirid/mortality24 (max |observed-mean| ~0.01; the only
        # non-zero column, categorical `sex`, is always observed so its decay
        # branch never fires). Fixed zero buffer; inject real means here if the
        # scaling ever changes.
        self.register_buffer("x_mean", torch.zeros(input_dim))

    def forward(self, value, mask, h, delta):
        """Advance one timestep.

        Args:
            value: (B, D) forward-filled value == last observation ``x'``.
            mask: (B, D) observed mask (1 == observed).
            h: (B, H) previous hidden state.
            delta: (B, D) time since last observation for this timestep.
        Returns:
            new hidden state (B, H).
        """
        gamma_x = torch.exp(-torch.relu(self.gamma_x_w * delta + self.gamma_x_b))
        gamma_h = torch.exp(-torch.relu(self.gamma_h_lin(delta)))

        # Input decay towards the empirical mean for missing entries; hidden decay.
        x_hat = mask * value + (1.0 - mask) * (gamma_x * value + (1.0 - gamma_x) * self.x_mean)
        h_decayed = gamma_h * h

        i_r, i_z, i_n = self.x2h(torch.cat([x_hat, mask], dim=-1)).chunk(3, dim=-1)
        h_r, h_z, h_n = self.h2h(h_decayed).chunk(3, dim=-1)

        # torch.nn.GRU formulation: candidate uses r ⊙ (U·ĥ).
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1.0 - z) * n + z * h_decayed
        return h_new


@gin.configurable
class GRUDNet(DLPredictionWrapper):
    """GRU-D model for irregularly sampled clinical time series.

    The first recurrent layer is a GRU-D cell that consumes (value, mask, delta)
    triplets; optional additional layers (``layer_dim`` > 1) are plain GRU layers
    stacked on top of the GRU-D hidden sequence, mirroring the multi-layer
    interface of :class:`~icu_benchmarks.models.dl_models.rnn.GRUNet`.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden_dim,
        layer_dim,
        num_classes,
        *args,
        feature_names=None,
        static_names=None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            feature_names=feature_names,
            static_names=static_names,
            dropout=dropout,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        num_features = input_size[2]
        dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx = self._build_feature_index(
            feature_names, num_features, static_names
        )
        # Buffers so the index mapping is saved/restored and device-moved with the model.
        self.register_buffer("dyn_idx", torch.tensor(dyn_idx, dtype=torch.long))
        self.register_buffer("dyn_mask_idx", torch.tensor(dyn_mask_idx, dtype=torch.long))
        self.register_buffer("sta_idx", torch.tensor(sta_idx, dtype=torch.long))
        self.register_buffer("sta_mask_idx", torch.tensor(sta_mask_idx, dtype=torch.long))

        # Dynamic and static value/mask pairs both feed the GRU-D decay cell; the
        # split is bookkeeping only (so static is identifiable) and does not change
        # the numerics -- the official GRU-D treats every column as a time series.
        decay_dim = len(dyn_idx) + len(sta_idx)
        logging.info(
            f"GRUDNet: {num_features} input features -> {len(dyn_idx)} dynamic + "
            f"{len(sta_idx)} static value/mask pairs (decay_dim={decay_dim})."
        )

        self.cell = GRUDCell(decay_dim, hidden_dim)
        self.stacked = (
            nn.GRU(hidden_dim, hidden_dim, layer_dim - 1, batch_first=True) if layer_dim > 1 else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.logit = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _build_feature_index(feature_names, num_features, static_names=None):
        """Pair each value column with its ``MissingIndicator_<col>`` mask column.

        The YAIB preprocessor adds a ``MissingIndicator`` to BOTH dynamic and static
        columns, so the naming convention alone cannot tell static (age/sex/...) value
        columns from dynamic ones. We therefore use the explicit ``static_names``
        (``vars["STATIC"]``, plumbed in from ``train_common``) to route static columns
        to their own index lists. Returns four feature-axis index lists: dynamic
        value/mask indices and static value/mask indices.
        """
        if feature_names is None:
            raise ValueError(
                "GRUDNet requires `feature_names` (the data column names) to pair value/mask "
                "columns. It is provided by icu_benchmarks.models.train.train_common."
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
                continue  # consumed as the partner of its value column
            partner = name_to_idx.get(MISSING_PREFIX + name, -1)
            if partner < 0:
                raise ValueError(f"Column {name} has no MissingIndicator partner; check the preprocessing pipeline.")
            if name in static_set:
                sta_idx.append(i)
                sta_mask_idx.append(partner)
            else:
                dyn_idx.append(i)
                dyn_mask_idx.append(partner)
        if not dyn_idx and not sta_idx:
            raise ValueError("GRUDNet found no value/mask column pairs; check the preprocessing pipeline.")
        return dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx

    def forward(self, x):
        # x: (B, T, F). Recover values (forward-filled == x') and observed masks for
        # dynamic and static columns, then feed them jointly to the GRU-D cell.
        batch_size, seq_len, _ = x.shape
        value = torch.cat([x.index_select(-1, self.dyn_idx), x.index_select(-1, self.sta_idx)], dim=-1)
        missing = torch.cat([x.index_select(-1, self.dyn_mask_idx), x.index_select(-1, self.sta_mask_idx)], dim=-1)
        mask = 1.0 - missing  # observed mask, 1 == observed

        h = x.new_zeros(batch_size, self.hidden_dim)
        delta = x.new_zeros(batch_size, value.size(-1))
        prev_mask = x.new_ones(batch_size, value.size(-1))  # delta stays 0 at t=0

        outputs = []
        for t in range(seq_len):
            m_t = mask[:, t, :]
            if t > 0:
                # Regular 1-hour grid: step is 1; accumulate while unobserved.
                delta = 1.0 + (1.0 - prev_mask) * delta
            h = self.cell(value[:, t, :], m_t, h, delta)
            outputs.append(h)
            prev_mask = m_t

        hidden_seq = torch.stack(outputs, dim=1)  # (B, T, H)
        if self.stacked is not None:
            h0 = hidden_seq.new_zeros(self.layer_dim - 1, batch_size, self.hidden_dim)
            hidden_seq, _ = self.stacked(hidden_seq, h0)
        if self.dropout is not None:
            hidden_seq = self.dropout(hidden_seq)
        pred = self.logit(hidden_seq)  # (B, T, num_classes)
        return pred
