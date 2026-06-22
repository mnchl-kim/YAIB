# -*- coding: utf-8 -*-
"""External validation of a trained YAIB model on a different target cohort.

Loads a YAIB run trained with the repeated nested CV protocol
(``repetition_{r}/fold_{f}/``) and evaluates it on a *different* target cohort,
keeping the same 5x5 nested-CV structure (fold-paired):

    source(rep r, fold f) model  ->  target(rep r, fold f) test split

Handles both model families with a single driver; the kind is auto-detected from
the source run's fold contents:
  - DL  (``model.ckpt`` / ``last.ckpt``): the framework's own eval path works, so
        we delegate to ``train_common(eval_only=True, load_weights=True,
        source_dir=...)`` which loads the Lightning checkpoint via
        ``load_from_checkpoint`` and runs ``trainer.test``, writing
        ``test_metrics.json`` exactly as a normal run does.
  - ML  (``model.joblib``): the native ``--eval`` path is broken for sklearn-style
        models (it calls LightningModule methods on a raw estimator and reuses one
        source dir for all folds), so we load the estimator and compute metrics
        directly, matching YAIB's ``test_metrics.json``:
            loss = log_loss(y, proba); AUC = roc_auc_score(y, proba[:, 1]);
            PR   = average_precision_score(y, proba[:, 1])

Common to both: the fold-pairing loop, per-fold target preprocessing (statistics
refit on each target fold), and the run-level result aggregation. The target is
preprocessed using the source run's ``train_config.gin`` (same vars / preprocessor
/ feature generation), so the feature matrix is built identically to training.

Usage:
    python scripts/external_validation.py \
        --source-run  /.../logs/hirid/mortality24/GRU/2026-05-19T12-46-39 \
        --target-dir  /.../YAIB-cohorts/data/grid_1hour/mortality24/miiv \
        --target-name miiv \
        --task        mortality24 \
        --log-dir     /.../YAIB/logs \
        --seed        1111
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import gin
import torch

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the
# RunMode enum / model class pickled inside YAIB Lightning checkpoints, so
# load_from_checkpoint() fails. The framework doesn't set weights_only; since
# these are our own trusted checkpoints, restore the pre-2.6 behaviour for this
# process only. (The native `--eval` path hits the same issue on torch>=2.6.)
# Harmless on the ML path: joblib.load does not go through torch.load.
_orig_torch_load = torch.load


def _torch_load_weights_false(*a, **k):
    # Force-override: Lightning's pl_load passes weights_only=True explicitly,
    # so setdefault is not enough — we must overwrite it.
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)


torch.load = _torch_load_weights_false

# Importing these modules registers their gin configurables (execute_repeated_cv,
# train_common, preprocess, dataset classes). The source train_config.gin also
# `import`s the model modules (GRUNet, BRITSNet, ...) when parsed, registering those.
from icu_benchmarks.constants import RunMode
from icu_benchmarks.cross_validation import execute_repeated_cv  # noqa: F401  (registers configurable)
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import aggregate_results, log_full_line, name_datasets

# ML eval path
import numpy as np
from joblib import load
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from icu_benchmarks.data.constants import DataSplit
from icu_benchmarks.data.loader import PredictionPolarsDataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="External validation of a trained YAIB model on a target cohort.")
    p.add_argument("--source-run", required=True, type=Path, help="Source run dir (contains repetition_*/fold_*/).")
    p.add_argument("--target-dir", required=True, type=Path, help="Target cohort parquet dir (dyn/sta/outc).")
    p.add_argument("--target-name", required=True, type=str, help="Target dataset name (e.g. miiv).")
    p.add_argument("--task", required=True, type=str, help="Task name, used for output paths (e.g. mortality24).")
    p.add_argument("--log-dir", required=True, type=Path, help="Root log dir (mirrors run.sh LOG_DIR).")
    p.add_argument("--seed", default=1111, type=int, help="Seed (must match the source run's split seed).")
    p.add_argument("--cv-repetitions", default=5, type=int)
    p.add_argument("--cv-folds", default=5, type=int)
    p.add_argument("--repetitions-to-eval", default=None, type=int, help="Limit repetitions (smoke test).")
    p.add_argument("--folds-to-eval", default=None, type=int, help="Limit folds per repetition (smoke test).")
    p.add_argument("--cpu", action="store_true", help="DL only: force CPU (default: use GPU if available).")
    return p


# Checkpoint/model file that marks a usable fold, per model kind.
_DL_CKPTS = ("model.ckpt", "last.ckpt")
_ML_MODELS = ("model.joblib", "last.joblib")


def detect_model_kind(source_run: Path) -> str:
    """Return 'dl' or 'ml' based on what the first source fold contains."""
    f0 = source_run / "repetition_0" / "fold_0"
    if any((f0 / c).exists() for c in _DL_CKPTS):
        return "dl"
    if any((f0 / c).exists() for c in _ML_MODELS):
        return "ml"
    raise ValueError(
        f"cannot detect model kind: none of {_DL_CKPTS + _ML_MODELS} found in {f0}"
    )


def _ml_model_path(src_fold: Path) -> Path:
    for c in _ML_MODELS:
        if (src_fold / c).exists():
            return src_fold / c
    raise FileNotFoundError(f"no ML model ({_ML_MODELS}) in {src_fold}")


def fold_has_model(src_fold: Path, kind: str) -> bool:
    if kind == "dl":
        return any((src_fold / c).exists() for c in _DL_CKPTS)
    return any((src_fold / c).exists() for c in _ML_MODELS)


def compute_metrics(model, rep: np.ndarray, labels: np.ndarray) -> dict:
    """Reproduce YAIB MLWrapper test metrics for binary classification."""
    proba = model.predict_proba(rep)
    pos = proba[:, 1]
    return {
        "loss": float(log_loss(labels, proba)),
        "AUC": float(roc_auc_score(labels, pos)),
        "PR": float(average_precision_score(labels, pos)),
    }


def eval_dl_fold(src_fold: Path, data, out_fold: Path, cpu: bool) -> None:
    """Delegate to the framework: load_from_checkpoint + trainer.test, which
    writes test_metrics.json into out_fold. All model/batch/trainer params come
    from the parsed source gin config."""
    train_common(
        data,
        log_dir=out_fold,
        eval_only=True,
        load_weights=True,
        source_dir=src_fold,
        mode=RunMode.classification,
        cpu=cpu,
    )


def maybe_bind_feature_names(sample_cfg: Path, data, target_name: str) -> None:
    """Work around models (BRITS, GRU-D) that require `feature_names` in __init__.

    Those models pair value columns with their MissingIndicator masks at
    construction time and store the resulting index as buffers (value_idx /
    mask_idx) in the checkpoint. But `feature_names` itself is NOT saved as a
    hyperparameter, so ``load_from_checkpoint`` (which rebuilds the model from
    hparams) calls __init__ with feature_names=None and raises.

    Since the index buffers are restored from state_dict anyway, __init__ only
    needs *some* correctly-ordered column list to pass. We rebuild it exactly as
    training does (train.py: train_dataset.get_feature_names() minus GROUP) from
    the target's train split — the source train_config.gin guarantees identical
    columns — and bind it via gin so it fills the missing hparam on reload.

    Models that don't take feature_names (GRU, ...) are left untouched.
    """
    m = re.search(r"train_common\.model\s*=\s*@(\w+)", sample_cfg.read_text())
    if not m:
        return
    cls_name = m.group(1)
    train_ds = PredictionPolarsDataset(data, split=DataSplit.train, name=target_name)
    feature_names = [c for c in train_ds.get_feature_names() if c != train_ds.vars["GROUP"]]
    try:
        with gin.unlock_config():
            gin.bind_parameter(f"{cls_name}.feature_names", feature_names)
        logging.info(f"bound {cls_name}.feature_names ({len(feature_names)} cols) for checkpoint reload")
    except Exception as e:
        # Configurable has no feature_names param (e.g. GRU) — nothing to do.
        logging.info(f"{cls_name} takes no feature_names; skip ({e})")


def eval_ml_fold(src_fold: Path, data, out_fold: Path, target_name: str) -> dict:
    """Load the sklearn estimator and compute metrics directly into out_fold."""
    test_ds = PredictionPolarsDataset(data, split=DataSplit.test, name=target_name)
    rep, labels, _ = test_ds.get_data_and_labels()

    model = load(_ml_model_path(src_fold))
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != rep.shape[1]:
        raise ValueError(
            f"feature mismatch: model expects {n_in}, target produced {rep.shape[1]}. "
            "Source and target preprocessing are not aligned."
        )

    metrics = compute_metrics(model, rep, labels)
    with open(out_fold / "test_metrics.json", "w") as fh:
        json.dump(metrics, fh, cls=JsonResultLoggingEncoder, indent=4)
    logging.info(
        f"n_test={len(labels)} feat={rep.shape[1]} "
        f"AUC={metrics['AUC']:.4f} PR={metrics['PR']:.4f} loss={metrics['loss']:.4f}"
    )
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    source_run = args.source_run.resolve()
    if not source_run.exists():
        raise ValueError(f"source run not found: {source_run}")
    sample_cfg = source_run / "repetition_0" / "fold_0" / "train_config.gin"
    if not sample_cfg.exists():
        raise ValueError(f"missing source train_config.gin: {sample_cfg}")

    kind = detect_model_kind(source_run)

    # The source train_config.gin is self-contained: its `import` lines register
    # the model configurables and it binds preprocess.* / dataset vars. Bindings
    # for configurables we don't use are skipped via skip_unknown=True. The DL
    # path needs finalize deferred (train_common finalizes) and a dataset rename
    # so logs use the target name instead of the source's.
    if kind == "dl":
        gin.parse_config_files_and_bindings([str(sample_cfg)], bindings=None, skip_unknown=True, finalize_config=False)
        name_datasets(args.target_name, args.target_name, args.target_name)
        seed_everything(args.seed, workers=True)
    else:
        gin.parse_config_files_and_bindings([str(sample_cfg)], bindings=None, skip_unknown=True)

    n_rep = args.repetitions_to_eval or args.cv_repetitions
    n_fold = args.folds_to_eval or args.cv_folds

    # Output dir mirrors run.sh layout: <log>/<target>/<task>/<MODEL>/<ts>_extval_from_<source>
    model_name = source_run.parents[0].name  # .../<MODEL>/<timestamp>
    source_name = source_run.parents[2].name  # .../<dataset>/<task>/<MODEL>/<timestamp>
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_root = args.log_dir / args.target_name / args.task / model_name / f"{ts}_extval_from_{source_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_full_line(f"External validation ({kind}) -> {out_root}", logging.INFO, char="=", num_newlines=1)
    logging.info(f"source={source_name}/{model_name}  target={args.target_name}  task={args.task}")
    logging.info(f"fold-paired over {n_rep} repetitions x {n_fold} folds (seed={args.seed})")

    overall_start = datetime.now()
    n_done = 0
    bound_feature_names = False
    for r in range(n_rep):
        for f in range(n_fold):
            src_fold = source_run / f"repetition_{r}" / f"fold_{f}"
            if not fold_has_model(src_fold, kind):
                logging.warning(f"skip: missing source model in {src_fold}")
                continue
            out_fold = out_root / f"repetition_{r}" / f"fold_{f}"
            out_fold.mkdir(parents=True, exist_ok=True)

            # Same preprocessing as a normal run; statistics refit on this target fold.
            t0 = datetime.now()
            data = preprocess_data(
                args.target_dir,
                seed=args.seed,
                cv_repetitions=args.cv_repetitions,
                repetition_index=r,
                cv_folds=args.cv_folds,
                fold_index=f,
                runmode=RunMode.classification,
            )
            preprocess_time = datetime.now() - t0

            # Columns are identical across folds; bind once for feature_names-requiring models.
            if kind == "dl" and not bound_feature_names:
                maybe_bind_feature_names(sample_cfg, data, args.target_name)
                bound_feature_names = True

            t1 = datetime.now()
            if kind == "dl":
                eval_dl_fold(src_fold, data, out_fold, cpu=args.cpu)
            else:
                eval_ml_fold(src_fold, data, out_fold, args.target_name)
            eval_time = datetime.now() - t1

            durations = {"preprocessing_duration": preprocess_time, "eval_duration": eval_time}
            with open(out_fold / "durations.json", "w") as fh:
                json.dump(durations, fh, cls=JsonResultLoggingEncoder)

            n_done += 1
            log_full_line(f"FINISHED rep{r}/fold{f} | preprocess {preprocess_time} | eval {eval_time}", logging.INFO)

    execution_time = datetime.now() - overall_start
    if n_done == 0:
        raise RuntimeError("no folds evaluated — check source run structure.")
    aggregate_results(out_root, execution_time)
    log_full_line(f"FINISHED external validation: {n_done} folds in {execution_time}", logging.INFO, char="=", num_newlines=1)
    logging.info(f"results -> {out_root}")
    return out_root


if __name__ == "__main__":
    main()
