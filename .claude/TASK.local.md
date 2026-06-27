# Working Task (local)

## Goal (stable)

Implement the **SeFT** baseline (Set Functions for Time Series, Horn et al. ICML 2020)
in the YAIB framework and validate on `hirid / mortality24`.
Part of the larger paper effort on clinical irregularly-sampled time-series modeling.

## References

- Paper: https://proceedings.mlr.press/v119/horn20a/horn20a.pdf (preprint https://arxiv.org/abs/1909.12064)
- Official code: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
  (`seft/models/deep_set_attention.py` → `DeepSetAttentionModel` / `SetAttentionLayer`)

## Key design decisions

- **No imputation / set view.** Rebuild the observation set from YAIB's
  (value, `MissingIndicator_<f>`) grid: a value is a real observation iff its mask says
  observed. Model never uses forward-filled values.
- **Dense candidate grid + masking** instead of ragged sets: candidates = (T × D_dynamic),
  `obs_mask` selects real ones. T=25, D=48 for hirid → N=1200, fine.
- **Static features = demographics** (from `vars["STATIC"]`): encoded once, prepended as one
  extra set element (matches official `demo_encoder`). Dynamic features = modalities.
- **Padded timesteps** (all-zero rows) detected and excluded from the set.
- **Whole-series (non-sequence) SeFT**: one prediction broadcast over T to fit the
  `(B,T,num_classes)` interface. Correct for mortality24 (label on last timestep only).
  Per-hour online tasks (aki/sepsis) would need the cumulative variant — out of scope here.
- Attention query `W_q` zero-init → starts as mean pooling (per paper). Expected: attention
  key-path params get zero grad on the very first step until `W_q` moves off zero.

## Files

- `icu_benchmarks/models/dl_models/seft.py` — `SeFTNet`, `SetAttentionLayer`, `PositionalEncoding`.
- `configs/prediction_models/SeFT.gin` — config; tuned: lr, phi_width, latent_width, rho_width,
  attn_dropout. Fixed to paper defaults: layers, n_heads=4, dot_prod_dim=128, psi_*, positional dims.
- `icu_benchmarks/models/__init__.py` — registered `SeFTNet`.
- `icu_benchmarks/models/train.py` — pass `feature_names` + `dataset_vars` to model ctor.
- `scripts/run.sh` — `-m seft` option + `PYTHONPATH` worktree fix.

## Status

- [x] Read codebase / run.sh / data / GRU-D reference.
- [x] Implement SeFTNet + config + wiring.
- [x] Unit forward/backward test passes (shapes, broadcast, grads, eval determinism).
- [x] Framework smoke test on hirid/mortality24 (1 rep × 5 folds, epochs=3, no tune) — exit 0.
      All folds produced test metrics: AUROC 0.652±0.012, AUPRC 0.138±0.007 (untrained-ish, 3 epochs).
      Log confirms split: 104 features -> 48 dynamic modalities + 4 static demographics, seq_len=25.
      (SHAP "cannot concat empty list" error is framework-generic when explain_features=False; harmless.)
- [x] Verify output metrics/logs (AUROC/AUPRC) — produced correctly.
- [x] Line-by-line fidelity review vs official `deep_set_attention.py` + `set_utils.py`.
      Fixed: rho decoder had one extra hidden layer (`_build_mlp` always appends a projection);
      now exactly n_rho_layers hidden + logit, matching `build_dense_dropout_model + Dense`.
      n_rho_layers default 2 -> 3 (paper default). Re-verified forward/backward.

## Next action

- Real experiment (full HPO): `scripts/run.sh -d hirid -t mortality24 -m seft -g <gpu> -j 16`
  (5×5 CV, --tune 30 calls). Then commit and check conflicts vs main.
