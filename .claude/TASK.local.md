# Local Task Tracking

## Goal (stable — do not change)

Write a top-tier ML/DL conference paper on clinical irregularly-sampled time-series modeling.
Current sub-task: implement the **GRU-D** baseline within YAIB and validate on `hirid / mortality24`.

## Reference

- GRU-D: Che et al., "Recurrent Neural Networks for Multivariate Time Series with Missing Values",
  Scientific Reports 8, 6085 (2018). Preprint: arXiv:1606.01865.
- Code: https://github.com/zhiyongc/GRU-D , https://github.com/PeterChe1990/GRU-D

## Key findings about the YAIB pipeline (verified, not assumed)

- Data reaching the model (hirid/mortality24) is `(B, T, F=104)`:
  - `[0:48]` dynamic values (scaled + forward-fill + zero-fill) → forward-fill makes this the GRU-D `x'` (last obs)
  - `[48:96]` `MissingIndicator_<dyn>` (1 == missing) → observed mask `m = 1 - this`
  - `[96:104]` static (age, sex, height, weight) + their MissingIndicator
  - Layout is reconstructed by name pairing (`<f>` ↔ `MissingIndicator_<f>`), so it is dataset/task agnostic.
- One label per stay (mortality24): loss is computed only at the last real timestep (others masked).
  Padding steps come after that step, so they do not affect the scored prediction.
- The installed `icu-benchmarks` console script imports the package from the **`paper`** worktree
  (editable install MAPPING). CWD is ignored by console scripts. Fix: `run.sh` now exports
  `PYTHONPATH="$YAIB_ROOT"` so each worktree uses its own code without touching the shared env.

## Implementation (done)

- `icu_benchmarks/models/dl_models/gru_d.py` — `GRUDCell` + `GRUDNet`.
  - δ computed internally on the uniform 1h grid from the mask; input decay `γ_x` (diagonal),
    hidden decay `γ_h` (full).
  - Empirical mean `x_mean`: FIXED zero buffer (not learnable). Verified that YAIB's `StepScale`
    == `StandardScaler(with_mean=True)` fit on observed values → observed mean = 0, so x̃ = 0 matches
    the paper exactly. (hirid check: max |observed-mean| ~0.01; only `sex` ~0.64 but it is always
    observed, so its decay branch never fires.) Assumption documented in `GRUDCell.__init__`.
  - Candidate gate uses the torch.nn.GRU form `r⊙(U·ĥ)` (per user request; single `h2h` of size 3H).
  - `layer_dim` is FIXED to 1 in GRUD.gin (`= 1`, not a tuple) → not tuned by HPO; verified that the
    model scope only tunes `hidden_dim`. The multi-layer stacking code path stays but is never exercised.
  - `_build_feature_index` splits columns into dynamic vs static value/mask pairs using `static_names`
    (`vars["STATIC"]`, plumbed from train_common), mirroring baseline_latentode. Both still feed the
    GRU-D cell as time series → bookkeeping only, numerically identical (value order [0..47,96..99]
    unchanged). Rationale: official GRU-D takes only (x, m, δ) and makes NO static/dynamic distinction
    (verified in Che's repo PeterChe1990/GRU-D + paper), so separation is purely for identifiability.
- `icu_benchmarks/models/__init__.py` — registered `GRUDNet` (import, `DLModel` Union, `__all__`).
- `icu_benchmarks/models/train.py` — pass `feature_names` (data cols, GROUP excluded) and
  `static_names` (`vars["STATIC"]`) to `model(...)`. Safe for all models (all accept `**kwargs`).
- `configs/prediction_models/GRUD.gin` — mirrors `GRU.gin` (same HPO space) with `@GRUDNet`; imports `gru_d`.
- `scripts/run.sh` — added `grud` model option (routes through GPU branch) + `PYTHONPATH` export.

## Status / Next action

- Unit test of forward/backward: PASS (pairing 52/52, out `(B,T,2)`, grads incl. `x_mean`).
- Smoke test on hirid/mortality24 (debug data, n_calls=1, epochs=2, GPU 0): PASS (exit 0).
  Full protocol structure produced: 5 reps × 5 folds, per-fold train/val/test metrics + aggregated/
  accumulated test metrics (loss/AUC/PR). Values are random-level (tiny debug data) as expected.
  Note: SHAP `test_shap_values` ERROR lines are pre-existing framework behavior (explain_features=False
  default) affecting ALL DL models, caught by try/except — not GRU-D specific, harmless.
- Next: commit this unit on the `baseline_grud` branch. Then run the FULL protocol
  (`scripts/run.sh -d hirid -t mortality24 -m grud -g <gpu> -j 8`) and record metrics.

## Run commands

- Smoke (fast): `PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0 icu-benchmarks train -d <cohorts>/mortality24/hirid -n hirid -t BinaryClassification -tn mortality24 -m GRUD --tune -db -hp tune_hyperparameters.n_calls=1 ... train_common.epochs=2 -l <log>`
- Full (protocol): `scripts/run.sh -d hirid -t mortality24 -m grud -g 0 -j 8`
