"""BRITS (Bidirectional Recurrent Imputation for Time Series), Cao et al., NeurIPS 2018.

Paper: https://arxiv.org/abs/1805.10572 / Official code: https://github.com/caow13/BRITS

YAIB integration:
- Value columns are paired with their ``MissingIndicator_<f>`` mask by name (as GRU-D);
  deltas are derived from the mask on the regular 1-hour grid. Static columns
  (``vars["STATIC"]``, plumbed from ``train_common``) are split out for identifiability
  but, as in GRU-D, still imputed/decayed as time series -- bookkeeping only.
- ``forward`` returns ``(per-timestep logits, aux_loss)``; the harness adds the
  classification loss to ``aux_loss`` (see ``DLPredictionWrapper.step_fn``), so
  ``aux_loss = impute_weight * (x_loss_f + x_loss_b) + consistency`` (BRITS label_weight==1).

Where the paper and the official repo disagree we follow the paper (also PyPOTS / this
project's GRU-D): ``beta`` is sigmoid-bounded, and ``delta_0=0`` using the previous mask.
"""

import logging
import math

import gin
import torch
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

MISSING_PREFIX = "MissingIndicator_"


class TemporalDecay(nn.Module):
    """Trainable decay ``gamma = exp(-relu(W*delta + b))``; ``diag`` for per-feature decay."""

    def __init__(self, input_dim: int, output_dim: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.empty(output_dim, input_dim))
        self.b = nn.Parameter(torch.empty(output_dim))
        if diag:
            assert input_dim == output_dim
            self.register_buffer("mask", torch.eye(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, delta):
        weight = self.W * self.mask if self.diag else self.W
        return torch.exp(-torch.relu(nn.functional.linear(delta, weight, self.b)))


class FeatureRegression(nn.Module):
    """Feature-based estimation with a zeroed diagonal (a feature cannot estimate itself)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, input_dim))
        self.b = nn.Parameter(torch.empty(input_dim))
        self.register_buffer("mask", torch.ones(input_dim, input_dim) - torch.eye(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, x):
        return nn.functional.linear(x, self.W * self.mask, self.b)


class RITS(nn.Module):
    """One-directional recurrent imputation. Returns per-timestep hidden states,
    the imputation sequence (for the consistency loss) and the accumulated imputation loss.
    """

    def __init__(self, decay_dim: int, hidden_dim: int):
        super().__init__()
        self.decay_dim = decay_dim
        self.hidden_dim = hidden_dim

        self.rnn_cell = nn.LSTMCell(decay_dim * 2, hidden_dim)
        self.temp_decay_h = TemporalDecay(decay_dim, hidden_dim, diag=False)
        self.temp_decay_x = TemporalDecay(decay_dim, decay_dim, diag=True)
        self.hist_reg = nn.Linear(hidden_dim, decay_dim)
        self.feat_reg = FeatureRegression(decay_dim)
        self.weight_combine = nn.Linear(decay_dim * 2, decay_dim)

    def forward(self, value, mask, delta):
        batch_size, seq_len, _ = value.shape
        h = value.new_zeros(batch_size, self.hidden_dim)
        c = value.new_zeros(batch_size, self.hidden_dim)

        x_loss = value.new_zeros(())
        hidden_seq = []
        imputations = []
        for t in range(seq_len):
            x = value[:, t, :]
            m = mask[:, t, :]

            gamma_h = self.temp_decay_h(delta[:, t, :])
            gamma_x = self.temp_decay_x(delta[:, t, :])
            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss = x_loss + torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1.0 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss = x_loss + torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            # sigmoid keeps beta in [0,1] per the paper; the official repo omits it.
            beta = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=-1)))
            c_h = beta * z_h + (1.0 - beta) * x_h
            x_loss = x_loss + torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1.0 - m) * c_h
            h, c = self.rnn_cell(torch.cat([c_c, m], dim=-1), (h, c))

            hidden_seq.append(h)
            imputations.append(c_c)

        return torch.stack(hidden_seq, dim=1), torch.stack(imputations, dim=1), x_loss


@gin.configurable
class BRITSNet(DLPredictionWrapper):
    """Forward and backward :class:`RITS` whose per-timestep logits are averaged and whose
    imputations are tied by a consistency loss; imputation + consistency returned as ``aux_loss``.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden_dim,
        num_classes,
        *args,
        feature_names=None,
        static_names=None,
        impute_weight: float = 0.3,
        consistency_weight: float = 0.1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            feature_names=feature_names,
            static_names=static_names,
            impute_weight=impute_weight,
            consistency_weight=consistency_weight,
            dropout=dropout,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.impute_weight = impute_weight
        self.consistency_weight = consistency_weight

        num_features = input_size[2]
        dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx = self._build_feature_index(
            feature_names, num_features, static_names
        )
        # Buffers so the index mapping moves with the model and is saved/restored.
        self.register_buffer("dyn_idx", torch.tensor(dyn_idx, dtype=torch.long))
        self.register_buffer("dyn_mask_idx", torch.tensor(dyn_mask_idx, dtype=torch.long))
        self.register_buffer("sta_idx", torch.tensor(sta_idx, dtype=torch.long))
        self.register_buffer("sta_mask_idx", torch.tensor(sta_mask_idx, dtype=torch.long))

        # Dynamic and static value/mask pairs are both imputed/decayed by RITS; the
        # split is bookkeeping only (so static is identifiable) and does not change the
        # numerics -- BRITS, like GRU-D, treats every column as a time series.
        decay_dim = len(dyn_idx) + len(sta_idx)
        logging.info(
            f"BRITSNet: {num_features} input features -> {len(dyn_idx)} dynamic + "
            f"{len(sta_idx)} static value/mask pairs (decay_dim={decay_dim})."
        )

        self.rits_f = RITS(decay_dim, hidden_dim)
        self.rits_b = RITS(decay_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        # Forward head named `logit` so the harness can read `self.logit.out_features`.
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.logit_b = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _build_feature_index(feature_names, num_features, static_names=None):
        """Pair each value column with its ``MissingIndicator_<col>`` mask column.

        The YAIB preprocessor adds a ``MissingIndicator`` to BOTH dynamic and static
        columns, so the naming convention alone cannot tell static (age/sex/...) value
        columns from dynamic ones. We use the explicit ``static_names``
        (``vars["STATIC"]``, plumbed from ``train_common``) to route static columns to
        their own index lists. Returns four feature-axis index lists: dynamic value/mask
        indices and static value/mask indices.
        """
        if feature_names is None:
            raise ValueError(
                "BRITSNet requires `feature_names` (the data column names) to pair value/mask "
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
                continue
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
            raise ValueError("BRITSNet found no value/mask column pairs; check the preprocessing pipeline.")
        return dyn_idx, dyn_mask_idx, sta_idx, sta_mask_idx

    @staticmethod
    def _deltas(mask):
        """delta_0 = 0; delta_t = 1 + (1 - m_{t-1}) * delta_{t-1} (paper; official uses m_t, delta_0=1)."""
        batch_size, seq_len, dim = mask.shape
        delta = mask.new_zeros(batch_size, seq_len, dim)
        for t in range(1, seq_len):
            delta[:, t, :] = 1.0 + (1.0 - mask[:, t - 1, :]) * delta[:, t - 1, :]
        return delta

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Recover values and observed masks for dynamic + static columns (both imputed).
        value = torch.cat([x.index_select(-1, self.dyn_idx), x.index_select(-1, self.sta_idx)], dim=-1)
        missing = torch.cat([x.index_select(-1, self.dyn_mask_idx), x.index_select(-1, self.sta_mask_idx)], dim=-1)
        mask = 1.0 - missing  # MissingIndicator==1 means missing

        h_f, imp_f, x_loss_f = self.rits_f(value, mask, self._deltas(mask))

        rev = torch.arange(seq_len - 1, -1, -1, device=x.device)
        value_b = value.index_select(1, rev)
        mask_b = mask.index_select(1, rev)
        h_b, imp_b, x_loss_b = self.rits_b(value_b, mask_b, self._deltas(mask_b))
        h_b = h_b.index_select(1, rev)
        imp_b = imp_b.index_select(1, rev)

        if self.dropout is not None:
            h_f = self.dropout(h_f)
            h_b = self.dropout(h_b)
        pred = 0.5 * (self.logit(h_f) + self.logit_b(h_b))

        consistency_loss = self.consistency_weight * torch.abs(imp_f - imp_b).mean()
        aux_loss = self.impute_weight * (x_loss_f + x_loss_b) + consistency_loss
        return pred, aux_loss
