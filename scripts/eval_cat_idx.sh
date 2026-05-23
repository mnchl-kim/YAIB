#!/bin/bash
# Evaluate a previously trained YAIB run on a cat_idx ablation data variant
# (e.g. cat_idx_0: only hr/resp/sbp/dbp/temp retained; other vars NaN-masked).
#
# Pairs each (i, j) inner CV iteration with the matching source repetition_i/fold_j
# (requires cross_validation.py source_dir pairing patch).
#
# Usage:
#   ./eval_cat_idx.sh -d DATASET -t TASK -m MODEL -r SOURCE_RUN_DIR \
#                     [-i CAT_IDX_SUBDIR] [-g GPU_ID] [-j NUM_THREADS] [-c START_CORE]
#
#   -d  DATASET            aumc | eicu | hirid | miiv
#   -t  TASK               mortality24 | aki | sepsis | kidney_function | los
#   -m  MODEL              lgbm | gru
#   -r  SOURCE_RUN_DIR     timestamp dir under logs/<dataset>/<task>/<model>/
#                          e.g. 2026-05-19T12-45-52
#   -i  CAT_IDX_SUBDIR     subdir under <cohort>/<task>/<dataset>/  (default: cat_idx_0)
#   -g  GPU_ID             integer GPU id (GRU only)                (default: 0)
#   -j  NUM_THREADS        CPU threads                              (default: 8)
#   -c  START_CORE         CPU core start, or 'auto'                (default: auto)
#
# Examples:
#   ./eval_cat_idx.sh -d hirid -t mortality24 -m lgbm -r 2026-05-19T12-45-52
#   ./eval_cat_idx.sh -d hirid -t mortality24 -m gru  -r 2026-05-19T12-46-39 -g 1
#
# Output layout (single new timestamp dir, 25 paired evaluations inside):
#   logs/<DATASET>/<TASK>/<MODEL_NAME>/<SOURCE_RUN_DIR>/<CAT_IDX_SUBDIR>/<new_ts>/
#         repetition_*/fold_*/test_metrics.json

set -euo pipefail

YAIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COHORTS_DATA=/team/team_bs_ic/personal/mincheol.kim/git/YAIB-cohorts/data
LOG_DIR=/team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs
SEED=1111

DATASET=""
TASK=""
MODEL=""
SOURCE_RUN=""
CAT_SUBDIR="cat_idx_0"
GPU_ID=0
NUM_THREADS=8
START_CORE="auto"

usage() {
  sed -n '2,29p' "${BASH_SOURCE[0]}" >&2
  exit 1
}

while getopts "d:t:m:r:i:g:j:c:h" opt; do
  case "$opt" in
    d) DATASET=$OPTARG ;;
    t) TASK=$OPTARG ;;
    m) MODEL=$OPTARG ;;
    r) SOURCE_RUN=$OPTARG ;;
    i) CAT_SUBDIR=$OPTARG ;;
    g) GPU_ID=$OPTARG ;;
    j) NUM_THREADS=$OPTARG ;;
    c) START_CORE=$OPTARG ;;
    h|*) usage ;;
  esac
done

[ -z "$DATASET" ]    && { echo "missing -d DATASET" >&2; usage; }
[ -z "$TASK" ]       && { echo "missing -t TASK" >&2; usage; }
[ -z "$MODEL" ]      && { echo "missing -m MODEL" >&2; usage; }
[ -z "$SOURCE_RUN" ] && { echo "missing -r SOURCE_RUN_DIR" >&2; usage; }

if ! [[ "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ]; then
  echo "ERROR: -j NUM_THREADS must be a positive int (got '$NUM_THREADS')" >&2; usage
fi
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: -g GPU_ID must be an int (got '$GPU_ID')" >&2; usage
fi
if [ "$START_CORE" != "auto" ] && ! [[ "$START_CORE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: -c START_CORE must be 'auto' or an int (got '$START_CORE')" >&2; usage
fi

case "$DATASET" in
  aumc|eicu|hirid|miiv) ;;
  *) echo "Unknown DATASET: $DATASET" >&2; usage ;;
esac

case "$TASK" in
  mortality24|aki|sepsis)
    TASK_TYPE=BinaryClassification
    LGBM_MODEL=LGBMClassifier
    ;;
  kidney_function|los)
    TASK_TYPE=Regression
    LGBM_MODEL=LGBMRegressor
    ;;
  *) echo "Unknown TASK: $TASK" >&2; usage ;;
esac

case "${MODEL,,}" in
  lgbm|lgbmclassifier|lgbmregressor) MODEL_NAME=$LGBM_MODEL ;;
  gru)                                MODEL_NAME=GRU ;;
  *) echo "Unknown MODEL: $MODEL (use: lgbm | gru)" >&2; usage ;;
esac

DATA_DIR="$COHORTS_DATA/$TASK/$DATASET/$CAT_SUBDIR"
SOURCE_BASE="$LOG_DIR/$DATASET/$TASK/$MODEL_NAME/$SOURCE_RUN"
TARGET_BASE="$LOG_DIR/$DATASET/$TASK/$MODEL_NAME/$SOURCE_RUN/$CAT_SUBDIR"

[ -d "$DATA_DIR" ]    || { echo "no data dir: $DATA_DIR" >&2; exit 1; }
[ -d "$SOURCE_BASE" ] || { echo "no source dir: $SOURCE_BASE" >&2; exit 1; }

# CPU core allocation (copied from run.sh) ------------------------------------
ALLOC_DIR=/tmp/yaib_cpu_alloc
mkdir -p "$ALLOC_DIR"
chmod 1777 "$ALLOC_DIR" 2>/dev/null || true

allocate_cpu_range() {
  local n=$1 total
  total=$(nproc)
  exec {fd}>"$ALLOC_DIR/.lock"
  flock -x "$fd"
  local f pid
  for f in "$ALLOC_DIR"/*.range; do
    [ -f "$f" ] || continue
    pid=$(basename "$f" .range)
    kill -0 "$pid" 2>/dev/null || rm -f "$f"
  done
  declare -A used
  local s e i
  for f in "$ALLOC_DIR"/*.range; do
    [ -f "$f" ] || continue
    while IFS='-' read -r s e; do
      for ((i=s; i<=e; i++)); do used[$i]=1; done
    done < "$f"
  done
  local found=-1
  for ((s=0; s<=total-n; s++)); do
    local ok=1
    for ((i=0; i<n; i++)); do
      if [ -n "${used[$((s+i))]:-}" ]; then ok=0; break; fi
    done
    if [ "$ok" = "1" ]; then found=$s; break; fi
  done
  if [ "$found" -lt 0 ]; then
    flock -u "$fd"
    echo "ERROR: cannot find $n free cores (total $total)" >&2
    exit 1
  fi
  echo "$found-$((found+n-1))" > "$ALLOC_DIR/$$.range"
  flock -u "$fd"
  echo "$found"
}
trap 'rm -f "$ALLOC_DIR/$$.range"' EXIT INT TERM

if [ "$START_CORE" = "auto" ]; then
  START_CORE=$(allocate_cpu_range "$NUM_THREADS")
fi
END_CORE=$((START_CORE + NUM_THREADS - 1))

# Env / thread caps -----------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yaib
export YAIB_PAIRED_SOURCE=1   # opt-in to per-(rep,fold) source model matching in cross_validation.py
export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
export POLARS_MAX_THREADS=$NUM_THREADS
export RAYON_NUM_THREADS=$NUM_THREADS
export NUMBA_NUM_THREADS=$NUM_THREADS

cd "$YAIB_ROOT"
mkdir -p "$TARGET_BASE"

RUNNER=""
command -v taskset >/dev/null && RUNNER="taskset -c ${START_CORE}-${END_CORE}"

# load_model() looks for model.joblib / model.ckpt, but training saves last.*.
# Add a symlink in every fold dir (idempotent).
ensure_load_alias() {
  local fold_dir=$1
  if [ "$MODEL_NAME" = "GRU" ]; then
    if [ ! -e "$fold_dir/model.ckpt" ] && [ -e "$fold_dir/last.ckpt" ]; then
      ln -s last.ckpt "$fold_dir/model.ckpt"
    fi
  else
    if [ ! -e "$fold_dir/model.joblib" ] && [ -e "$fold_dir/last.joblib" ]; then
      ln -s last.joblib "$fold_dir/model.joblib"
    fi
  fi
}

# Pre-create symlinks for every fold dir (the inner CV loop loads each one).
shopt -s nullglob
for fold_dir in "$SOURCE_BASE"/repetition_*/fold_*; do
  [ -f "$fold_dir/train_config.gin" ] || continue
  ensure_load_alias "$fold_dir"
done
shopt -u nullglob

# Entry point used by run.py for train_config.gin parsing. cross_validation.py
# will swap it to the matching (repetition_i, fold_j) per inner iteration.
ENTRY_FOLD="$SOURCE_BASE/repetition_0/fold_0"
[ -f "$ENTRY_FOLD/train_config.gin" ] || {
  echo "no train_config.gin at $ENTRY_FOLD" >&2; exit 1; }

# run.py always writes its run_dir to:
#   <log_dir>/<name>/<task_name>/<model>/_from_<source_name>/<timestamp>
# We can't reshape that path via flags, so we mv it after the run.
AUTO_PARENT="$LOG_DIR/$CAT_SUBDIR/$TASK/$MODEL_NAME/_from_$DATASET"

echo "============================================================"
echo " EVAL  data    = $DATA_DIR"
echo "       source  = $SOURCE_BASE  (paired diagonal)"
echo "       target  = $TARGET_BASE"
echo "       model   = $MODEL_NAME"
echo "       threads = $NUM_THREADS  cores = ${START_CORE}-${END_CORE}"
[ "$MODEL_NAME" = "GRU" ] && echo "       gpu     = $GPU_ID"
echo "============================================================"

if [ "$MODEL_NAME" = "GRU" ]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  $RUNNER python "$YAIB_ROOT/scripts/_run_local.py" \
    -d "$DATA_DIR" \
    -n "$CAT_SUBDIR" \
    -t "$TASK_TYPE" \
    -tn "$TASK" \
    -m "$MODEL_NAME" \
    -s "$SEED" \
    -l "$LOG_DIR/" \
    --eval \
    --source-dir "$ENTRY_FOLD" \
    -sn "$DATASET"
else
  CUDA_VISIBLE_DEVICES="" \
  $RUNNER python "$YAIB_ROOT/scripts/_run_local.py" \
    -d "$DATA_DIR" \
    -n "$CAT_SUBDIR" \
    -t "$TASK_TYPE" \
    -tn "$TASK" \
    -m "$MODEL_NAME" \
    -s "$SEED" \
    -l "$LOG_DIR/" \
    --eval \
    --source-dir "$ENTRY_FOLD" \
    -sn "$DATASET" \
    --cpu
fi

# Move the newly created timestamp dir to the requested target layout.
new_ts=$(ls -1t "$AUTO_PARENT" 2>/dev/null | head -1 || true)
if [ -n "$new_ts" ] && [ -d "$AUTO_PARENT/$new_ts" ]; then
  mv "$AUTO_PARENT/$new_ts" "$TARGET_BASE/$new_ts"
  echo "moved -> $TARGET_BASE/$new_ts"
else
  echo "warn: no new timestamp dir found under $AUTO_PARENT" >&2
fi

# Cleanup auto-created empty parents.
rmdir "$AUTO_PARENT" \
      "$LOG_DIR/$CAT_SUBDIR/$TASK/$MODEL_NAME" \
      "$LOG_DIR/$CAT_SUBDIR/$TASK" \
      "$LOG_DIR/$CAT_SUBDIR" 2>/dev/null || true

echo
echo "[done] eval logs under: $TARGET_BASE"
