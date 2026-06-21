#!/bin/bash
# YAIB external-validation runner: evaluate a model trained on one cohort
# fold-paired on a different target cohort (same 5x5 nested-CV structure).
# See external_validation.py.
#
# Single runner for both model families. The kind is auto-detected from the
# source run's fold contents:
#   - DL  (model.ckpt / last.ckpt): runs on GPU; the given GPU is exposed via
#         CUDA_VISIBLE_DEVICES. Use --cpu to force CPU.
#   - ML  (model.joblib):          CPU-only; GPU is hidden. -g is accepted but
#         ignored (kept for a uniform interface).
#
# Usage:
#   ./run_extval.sh -S SOURCE_RUN -d TARGET_DATASET -t TASK [-g GPU_ID] [-j NUM_THREADS] [-c START_CORE] [-R N] [-F N] [--cpu]
#
#   -S  SOURCE_RUN     source run dir (.../<dataset>/<task>/<MODEL>/<timestamp>)
#   -d  TARGET_DATASET aumc | eicu | hirid | miiv      (target cohort)
#   -t  TASK           mortality24 | aki | sepsis | kidney_function | los
#   -g  GPU_ID         GPU id (DL only; ignored for ML)  (default: 0)
#   -j  NUM_THREADS    CPU thread number                 (default: 8)
#   -c  START_CORE     CPU core start number or 'auto'   (default: auto)
#   -R  REPS           repetitions to eval (smoke test)  (default: all=5)
#   -F  FOLDS          folds per repetition (smoke test) (default: all=5)
#       --cpu          DL only: evaluate on CPU instead of GPU (slow)
#
# Example (smoke test: 1 rep x 1 fold, 8 threads, gpu 7):
#   ./run_extval.sh -S /.../logs/hirid/mortality24/GRU/2026-05-19T12-46-39 \
#                   -d miiv -t mortality24 -g 7 -j 8 -R 1 -F 1

set -euo pipefail

# ======================== 고정 경로 ========================
YAIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COHORTS_DATA=/data1/mincheol.kim/YAIB-cohorts/data/grid_1hour
LOG_DIR=/team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs
SEED=1111
# ==========================================================

SOURCE_RUN=""
DATASET=""
TASK=""
GPU_ID=0
NUM_THREADS=8
START_CORE="auto"
REPS=""
FOLDS=""
USE_CPU=0

usage() {
  cat <<USAGE >&2
Usage: $0 -S SOURCE_RUN -d TARGET_DATASET -t TASK [-g GPU_ID] [-j NUM_THREADS] [-c START_CORE] [-R REPS] [-F FOLDS] [--cpu]

  -S  SOURCE_RUN     source run dir (.../<dataset>/<task>/<MODEL>/<timestamp>)
  -d  TARGET_DATASET aumc | eicu | hirid | miiv
  -t  TASK           mortality24 | aki | sepsis | kidney_function | los
  -g  GPU_ID         GPU id (DL only; ignored for ML)  (default: 0)
  -j  NUM_THREADS    CPU thread number                 (default: 8)
  -c  START_CORE     CPU core start number or 'auto'   (default: auto)
  -R  REPS           repetitions to eval               (default: all)
  -F  FOLDS          folds per repetition              (default: all)
      --cpu          DL only: evaluate on CPU
USAGE
  exit 1
}

# Long option --cpu 추출 후 getopts
ARGS=()
for a in "$@"; do
  case "$a" in
    --cpu) USE_CPU=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]}"

while getopts "S:d:t:g:j:c:R:F:h" opt; do
  case "$opt" in
    S) SOURCE_RUN=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    t) TASK=$OPTARG ;;
    g) GPU_ID=$OPTARG ;;
    j) NUM_THREADS=$OPTARG ;;
    c) START_CORE=$OPTARG ;;
    R) REPS=$OPTARG ;;
    F) FOLDS=$OPTARG ;;
    h|*) usage ;;
  esac
done

[ -z "$SOURCE_RUN" ] && { echo "missing -S SOURCE_RUN" >&2; usage; }
[ -z "$DATASET" ]    && { echo "missing -d TARGET_DATASET" >&2; usage; }
[ -z "$TASK" ]       && { echo "missing -t TASK" >&2; usage; }

if ! [[ "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ]; then
  echo "ERROR: -j NUM_THREADS 는 양의 정수여야 합니다 (받은 값: '$NUM_THREADS')" >&2; usage
fi
if [ "$START_CORE" != "auto" ] && ! [[ "$START_CORE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: -c START_CORE 는 'auto' 또는 정수여야 합니다 (받은 값: '$START_CORE')" >&2; usage
fi

case "$DATASET" in
  aumc|eicu|hirid|miiv) ;;
  *) echo "잘못된 TARGET_DATASET: $DATASET" >&2; usage ;;
esac
case "$TASK" in
  mortality24|aki|sepsis|kidney_function|los) ;;
  *) echo "잘못된 TASK: $TASK" >&2; usage ;;
esac

[ -d "$SOURCE_RUN" ] || { echo "ERROR: source run 없음: $SOURCE_RUN" >&2; exit 1; }
if [ ! -d "$COHORTS_DATA/$TASK/$DATASET" ]; then
  echo "ERROR: 코호트 폴더 없음: $COHORTS_DATA/$TASK/$DATASET" >&2; exit 1
fi

# ---------------- 모델 종류 자동 감지 (DL=ckpt, ML=joblib) ----------------
F0="$SOURCE_RUN/repetition_0/fold_0"
if [ -f "$F0/model.ckpt" ] || [ -f "$F0/last.ckpt" ]; then
  MODEL_KIND="dl"
elif [ -f "$F0/model.joblib" ]; then
  MODEL_KIND="ml"
else
  echo "ERROR: 모델 종류 감지 실패 (model.ckpt/last.ckpt/model.joblib 없음): $F0" >&2; exit 1
fi

# DL(GPU 사용)일 때만 GPU_ID 검증. ML이거나 --cpu면 불필요.
if [ "$MODEL_KIND" = "dl" ] && [ "$USE_CPU" = "0" ]; then
  if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: -g GPU_ID 는 정수여야 합니다 (받은 값: '$GPU_ID')" >&2; usage
  fi
fi

# ---------------- CPU core 자동 할당 (run.sh와 동일 메커니즘) ----------------
ALLOC_DIR=/tmp/yaib_cpu_alloc
mkdir -p "$ALLOC_DIR"
chmod 1777 "$ALLOC_DIR" 2>/dev/null || true

allocate_cpu_range() {
  local n=$1
  local total; total=$(nproc)
  exec {fd}>"$ALLOC_DIR/.lock"; flock -x "$fd"
  local f pid
  for f in "$ALLOC_DIR"/*.range; do
    [ -f "$f" ] || continue
    pid=$(basename "$f" .range)
    kill -0 "$pid" 2>/dev/null || rm -f "$f"
  done
  declare -A used; local s e i
  for f in "$ALLOC_DIR"/*.range; do
    [ -f "$f" ] || continue
    while IFS='-' read -r s e; do for ((i=s; i<=e; i++)); do used[$i]=1; done; done < "$f"
  done
  local found=-1
  for ((s=0; s<=total-n; s++)); do
    local ok=1
    for ((i=0; i<n; i++)); do if [ -n "${used[$((s+i))]:-}" ]; then ok=0; break; fi; done
    if [ "$ok" = "1" ]; then found=$s; break; fi
  done
  if [ "$found" -lt 0 ]; then flock -u "$fd"; echo "ERROR: 빈 코어 $n개 못 찾음 (총 $total)" >&2; exit 1; fi
  echo "$found-$((found+n-1))" > "$ALLOC_DIR/$$.range"
  flock -u "$fd"; echo "$found"
}
cleanup_alloc() { rm -f "$ALLOC_DIR/$$.range"; }
trap cleanup_alloc EXIT INT TERM

if [ "$START_CORE" = "auto" ]; then START_CORE=$(allocate_cpu_range "$NUM_THREADS"); fi
END_CORE=$((START_CORE + NUM_THREADS - 1))

# ---------------- env / cwd ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yaib

export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
export POLARS_MAX_THREADS=$NUM_THREADS
export RAYON_NUM_THREADS=$NUM_THREADS
export NUMBA_NUM_THREADS=$NUM_THREADS

cd "$YAIB_ROOT"

# Use THIS worktree's code, not whatever the shared editable install points to.
export PYTHONPATH="$YAIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUNNER=""
if command -v taskset >/dev/null 2>&1; then RUNNER="taskset -c ${START_CORE}-${END_CORE}"; fi

EXTRA=""
[ -n "$REPS" ]  && EXTRA="$EXTRA --repetitions-to-eval $REPS"
[ -n "$FOLDS" ] && EXTRA="$EXTRA --folds-to-eval $FOLDS"

# device 결정: DL은 지정 GPU 노출(--cpu면 숨김), ML은 항상 숨김(CPU-only).
if [ "$MODEL_KIND" = "dl" ] && [ "$USE_CPU" = "0" ]; then
  VIS="$GPU_ID"
  DEV_DESC="gpu $GPU_ID"
else
  VIS=""
  DEV_DESC=$([ "$MODEL_KIND" = "ml" ] && echo "cpu (ML)" || echo "cpu")
fi
[ "$MODEL_KIND" = "dl" ] && [ "$USE_CPU" = "1" ] && EXTRA="$EXTRA --cpu"

echo "============================================================"
echo " EXTERNAL VALIDATION (${MODEL_KIND})"
echo " source=$SOURCE_RUN"
echo " target=$DATASET  task=$TASK"
echo " threads=$NUM_THREADS  cores=${START_CORE}-${END_CORE}  device=${DEV_DESC}"
echo " reps=${REPS:-all}  folds=${FOLDS:-all}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$VIS" \
$RUNNER python scripts/external_validation.py \
  --source-run "$SOURCE_RUN" \
  --target-dir "$COHORTS_DATA/$TASK/$DATASET" \
  --target-name "$DATASET" \
  --task "$TASK" \
  --log-dir "$LOG_DIR" \
  --seed "$SEED" \
  $EXTRA
