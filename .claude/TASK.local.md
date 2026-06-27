# Task (local working notes)

## Goal (stable)

Implement the **mTAND** baseline in YAIB and validate on `hirid / mortality24`.

- Paper: https://arxiv.org/pdf/2101.10318 (ICLR 2021, "Multi-Time Attention Networks for
  Irregularly Sampled Time Series")
- Official code: https://github.com/reml-lab/mTAN (`multiTimeAttention`, `enc_mtan_classif`)

## Design decisions

- Variant = **mTAND-Full** (the paper's PhysioNet/MIMIC mortality variant;
  `tan_classification.py --enc mtan_rnn --dec mtan_rnn`): VAE mTAND encoder
  (`EncMtanRNN`) -> latent z0 at reference points; `CreateClassifier` (GRU+MLP) for
  the label; `DecMtanRNN` reconstructs observations. Objective = recon ELBO + alpha*CE.
- YAIB integration: forward returns `(logits_broadcast(B,T,C), aux_loss)`. `step_fn`
  (wrappers.py:330) adds `aux_loss` to its balanced CE, so `aux_loss = recon_ELBO/alpha`
  reproduces the official objective up to a global scale. learn_emb=True (kept per user).
  k_iwae=1 default (exact official path); >1 uses IWAE bound for recon, sample 0 for clf.
- Value/mask pairs recovered by name (`_build_feature_index`, as GRU-D); leftover columns
  -> static, concatenated to the classifier GRU summary.
- Official defaults matched: latent_dim=32, gen_hidden=50, embed_time=128, num_ref=128,
  num_heads=1, alpha=100, std=0.01, kl_coef=1 (no --kl/--norm in the mortality command).

### Diff vs official (reviewed against reml-lab/mTAN @ master)
- Faithful: multiTimeAttention, enc/dec/classifier modules, time embedding, recon ELBO
  (log_normal_pdf + normal_kl + IWAE), input = concat(values, mask).
- Intentional adaptations: (B,T,C) broadcast for YAIB per-step wrapper; aux_loss=recon/alpha
  (wrapper owns CE, with balanced class weights); time grid = arange(T)/(T-1) in [0,1]
  (official uses observed_tp/48); HPO over hidden_dim/latent_dim/lr; static concat path.

## Key facts about data / pipeline

- `hirid/mortality24`: every stay is exactly **T=25** hourly steps (0..24h) → **no padding**.
- Single label per stay at the **last** timestep; other steps masked out by `step_fn`.
- `generate_features=False`: each value `<f>` paired with `MissingIndicator_<f>` (1==missing),
  forward-fill + zero-fill. Static (age/sex/height/weight) also get `MissingIndicator_*`.
  Features standard-scaled (observed mean ~0). Time grid normalized to [0,1].

## Progress

- [x] Read codebase, run.sh, data, GRU-D reference.
- [x] v1: implemented mTAND-Enc (enc_mtan_classif). Committed 8f8a20c.
- [x] Reviewed against official code; user chose mTAND-Full + learn_emb=True.
- [x] Rewrote `mtand.py` to mTAND-Full (EncMtanRNN/DecMtanRNN/CreateClassifier + IWAE ELBO).
      Class name `MTANDNet` kept (no gin/__init__/run.sh churn).
- [x] Updated `mTAND.gin` (latent_dim/gen_hidden/alpha/k_iwae/std; tune hidden_dim/latent_dim/lr).
- [x] `train.py` feature_names + `run.sh` (mtand + PYTHONPATH) — unchanged from v1.
- [x] Unit test: forward returns (pred(B,T,C), aux); enc/dec/clf all get grads; k_iwae 1 & 3.
- [x] Smoke test hirid/mortality24 (--no-tune, 2 epochs, GPU 4): 104 cols -> 52 pairs;
      enc 73.6K/dec 99.7K/clf 122K params; trains + CV + metrics, no nan/inf. Loss dominated
      by recon ELBO (expected for std=0.01; AUC meaningless at 2 epochs/random lr).
- [x] Commit mTAND-Full (2333b20).
- [x] First tuned run (logs .../mTAND/2026-06-03T19-03-46): severe overfit — train AUC 0.999,
      test AUC 0.736 (LGBM ~0.850). Chosen HP hidden_dim=247, latent_dim=111.
- [x] Root cause: (a) HPO/early-stop selected on joint loss (recon-dominated, ~263) not
      classification; (b) search ranges (hidden up to 512, latent up to 128) >> paper grid
      {32,64,128} -> oversized model memorizes.
- [x] Reviewed paper HP spec (ar5iv): ref=128, embed=128 fixed; tune latent & rec_hidden over
      {32,64,128}; num_heads {1,2,4}; gen_hidden=50; lr=1e-4; alpha (lambda) 100/5 per dataset;
      std=0.01; k_iwae=1; **KL annealing rate 0.99 improved perf**; select on val AUC.
- [x] Fixes applied:
      * wrappers.py: split `clf_loss` (pure CE) from `loss` (CE + aux); log both.
      * train.py: EarlyStopping on `val/clf_loss`; HPO objective `test/clf_loss`; plumb
        `static_names` (vars["STATIC"]).
      * mtand.py: route static cols (age/sex/height/weight) to classifier branch (their
        MissingIndicator partners dropped); **KL annealing** (kl_anneal/kl_rate=0.99/kl_wait=10
        via self.current_epoch).
      * mTAND.gin (user-edited): hidden_dim/latent_dim (16,128,"log"); num_heads [1,2,4].
- [x] Unit + smoke test (GPU 7, 14 epochs): clf_loss logged separately, KL annealing runs,
      static routing OK, num_heads categorical OK, no errors.

## Run 2026-06-23T19-22-20 result (fixes applied)

- test AUC mean **0.8011** ± 0.027 (n=25), best fold 0.856; target = LGBM/GRU ~0.850.
- clf_loss healthy (train 0.43 / val 0.53). recon ELBO trains fine (train/loss 12527->1816
  monotonic, train~=val) — large absolute value is just std=0.01 (5000x squared-err scale,
  summed over observed cells), NOT a bug.
- Diagnosis of the 0.05 gap to 0.85:
  1. Chosen HP hit the search ceiling: hidden_dim=128 AND latent_dim=128 (range was (16,128)).
     -> capacity-capped.
  2. High seed variance (rep4 0.839 vs others 0.78-0.80, same HP) -> training instability,
     driven by recon gradient dominating the joint loss ~99.98% (1816 vs 0.43 at alpha=100).

## Official HP comparison (paper + reml-lab/mTAN @ master, mortality commands)

PhysioNet: `--alpha 100 --lr 1e-4 --batch-size 50 --rec-hidden 256 --gen-hidden 50
--latent-dim 20 --enc/dec mtan_rnn --k-iwae 1 --std 0.01 --classif --norm --kl --learn-emb`
MIMIC: same but `--alpha 5 --latent-dim 128 --batch-size 128`, no `--kl`.

KEY FINDING: both commands use **`--norm`**, which in `utils.compute_losses` does
`logpx /= observed_mask.sum(-1).sum(-1); analytic_kl /= same`. Our impl had NO normalization
(summed) -> recon ~1e5 dwarfed CE (~99.98% of gradient). The earlier alpha-upward idea was
just compensating for the missing norm; the official fix is `--norm` + alpha in 5..100.

## DONE: implemented `--norm`

- mtand.py: added `norm: bool = True` ctor param (+ super kwargs, `self.norm`, docstrings).
  `_reconstruction_elbo` now divides logpx & analytic_kl by `mask.sum(-1).sum(-1).clamp(min=1)`
  when `self.norm`. Fixed the (factually wrong) docstring that claimed --norm was off.
- CPU forward check (norm off vs on): aux_loss 2500 -> 50.7 (~50x, == observed-cell count),
  both finite, shapes/value/static routing OK.
- gin (user-edited): hidden_dim (64,512,log), latent_dim (16,256,log), num_heads [1,2,4],
  alpha (5,1000,log) -- now correctly calibrated for normalized recon.

## Next action

Smoke test ON HOLD per user. When cleared, run (ASK GPU id + thread budget first; do NOT
broad-pkill): `scripts/run.sh -d hirid -t mortality24 -m mtand -g <gpu> -j <threads>`
Watch: loss magnitude now small (recon ~per-obs), clf_loss-based selection, train-vs-test gap,
and whether best alpha lands in the official 5..100 region. Target test AUC ~0.85.
Remaining minor diffs vs official: num_heads search (official fixes 1); batch 64 vs 50/128.
