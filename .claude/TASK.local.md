# Local Task Tracking

## Goal (stable)

Implement the **BRITS** baseline in YAIB and validate on `hirid / mortality24`.

- Paper: https://arxiv.org/pdf/1805.10572 (NeurIPS 2018)
- Official code: https://github.com/caow13/BRITS

## Branch

`baseline_brits` (worktree). Template to mirror: `baseline_grud` branch (GRU-D).

## Key findings

- YAIB step_fn (`wrappers.py:327-333`): if `forward` returns a tuple `(out, aux_loss)`,
  `aux_loss` is added to the classification loss. → use it for BRITS imputation + consistency loss.
- Data layout (preprocessor.py): every value column `<f>` is StandardScaled (observed mean≈0),
  paired with `MissingIndicator_<f>` (1==missing), then forward-fill + zero-fill.
  Static (age/sex/height/weight) also get MissingIndicator. → reuse GRU-D `_build_feature_index` pairing.
- Regular 1h grid: delta_t = 1 + (1-m_{t-1})*delta_{t-1}, delta_0=0 (per direction).
- Prediction is per-timestep `(B, T, num_classes)`; step_fn masked_select with padding mask.
- `train.py` must pass `feature_names` (GROUP excluded) — not yet in this branch; need to add (same as grud).

## Plan / Progress

- [x] Read codebase, run.sh, data structure, GRU-D template.
- [x] Fetch official BRITS code (rits.py / brits.py).
- [x] Implement `icu_benchmarks/models/dl_models/brits.py` (TemporalDecay, FeatureRegression, RITS, BRITSNet).
- [x] Add `configs/prediction_models/BRITS.gin`.
- [x] Register `BRITSNet` in `icu_benchmarks/models/__init__.py`.
- [x] Patch `train.py` (feature_names) and `scripts/run.sh` (brits option + PYTHONPATH).
- [x] Smoke test on hirid/mortality24 — full CV pipeline runs end-to-end; metrics (loss/AUC/PR) produced.
      (`-db` debug + 2 epochs gives random-level AUC, as expected.)
- [x] Commit.

## Paper vs. official-code review (decided: follow PAPER)

Verified line-by-line against caow13/BRITS (rits.py, brits.py, input_process.py).
Two paper-vs-code discrepancies; user chose to keep the **paper-faithful** version (also matches PyPOTS + project GRU-D):

1. beta combine: paper `sigmoid(W[gamma_x;m]+b)` (∈[0,1]); official omits sigmoid. → we keep sigmoid.
2. delta: paper `delta_0=0, delta_t=1+(1-m_{t-1})delta_{t-1}` (prev mask); official `delta_0=1`, current mask. → we keep paper.

Documented inline in brits.py (module docstring + RITS.forward + _deltas).
Everything else matches official exactly; remaining differences are unavoidable YAIB-harness adaptations
(per-timestep CrossEntropy head vs. sequence-level BCE; averaged logits vs. averaged probabilities; dynamic seq/feat dims).

## Notes

- Forward returns `(pred (B,T,num_classes), aux_loss)`; `aux_loss = impute_weight*(x_loss_f+x_loss_b) + consistency`.
  Classification loss added by harness step_fn.
- Forward head named `self.logit` (harness reads `self.logit.out_features` for metric selection); backward = `self.logit_b`.
- Static (age/sex/height/weight) carry MissingIndicator masks too → all features paired; static_idx empty in practice.
- SHAP "Failed to save shap values" warning is pre-existing framework behavior (affects GRU-D identically), harmless.
- For real runs use GPU via `scripts/run.sh -m brits` (single GPU avoids the CPU/DDP multi-rank test error).

## Next action

Real run: `scripts/run.sh -d hirid -t mortality24 -m brits -g <gpu> -j 16` (full --tune CV).
