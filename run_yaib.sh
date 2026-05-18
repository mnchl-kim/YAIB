#!/bin/bash
# YAIB paper reproduction runner
#
# Usage:
#   ./run_yaib.sh -d DATASET -t TASK [-m MODEL] [-g GPU_ID] [-j NUM_THREADS] [-c START_CORE]
#
#   -d  DATASET       aumc | eicu | hirid | miiv
#   -t  TASK          mortality24 | aki | sepsis | kidney_function | los
#   -m  MODEL         lgbm | gru | both          (default: both)
#   -g  GPU_ID        정수 (GRU에만 사용)         (default: 0)
#   -j  NUM_THREADS   CPU thread 수               (default: 8)
#   -c  START_CORE    CPU 코어 시작 번호          (default: auto)
#                     auto: 빈 코어 자동 할당     (다른 인스턴스와 충돌 안 함)
#                     N:   N ~ N+NUM_THREADS-1 사용
#
# Examples:
#   ./run_yaib.sh -d miiv -t mortality24                  # 코어 자동 할당
#   ./run_yaib.sh -d eicu -t aki -m gru -g 1 -j 4         # 4 threads, 자동 코어
#   ./run_yaib.sh -d hirid -t los -m lgbm -j 8 -c 16      # 코어 16~23 수동

set -euo pipefail

# ======================== 고정 경로 ========================
YAIB_ROOT=/team/team_bs_ic/personal/mincheol.kim/git/YAIB/paper
COHORTS_DATA=/team/team_bs_ic/personal/mincheol.kim/git/YAIB-cohorts/data
LOG_DIR=/team/team_bs_ic/personal/mincheol.kim/git/yaib_logs
SEED=1111
# ==========================================================

DATASET=""
TASK=""
MODEL=""
GPU_ID=0
NUM_THREADS=8
START_CORE="auto"

usage() {
  cat <<USAGE >&2
Usage: $0 -d DATASET -t TASK [-m MODEL] [-g GPU_ID] [-j NUM_THREADS] [-c START_CORE]

  -d  DATASET       aumc | eicu | hirid | miiv
  -t  TASK          mortality24 | aki | sepsis | kidney_function | los
  -m  MODEL         lgbm | gru | both          (default: both)
  -g  GPU_ID        정수 (GRU에만 사용)         (default: 0)
  -j  NUM_THREADS   CPU thread 수               (default: 8)
  -c  START_CORE    CPU 코어 시작 번호 또는 'auto' (default: auto)
USAGE
  exit 1
}

while getopts "d:t:m:g:j:c:h" opt; do
  case "$opt" in
    d) DATASET=$OPTARG ;;
    t) TASK=$OPTARG ;;
    m) MODEL=$OPTARG ;;
    g) GPU_ID=$OPTARG ;;
    j) NUM_THREADS=$OPTARG ;;
    c) START_CORE=$OPTARG ;;
    h|*) usage ;;
  esac
done

[ -z "$DATASET" ] && { echo "missing -d DATASET" >&2; usage; }
[ -z "$TASK" ]    && { echo "missing -t TASK"    >&2; usage; }

# 숫자 옵션 검증
if ! [[ "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ]; then
  echo "ERROR: -j NUM_THREADS 는 양의 정수여야 합니다 (받은 값: '$NUM_THREADS')" >&2
  usage
fi
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: -g GPU_ID 는 정수여야 합니다 (받은 값: '$GPU_ID')" >&2
  usage
fi
if [ "$START_CORE" != "auto" ] && ! [[ "$START_CORE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: -c START_CORE 는 'auto' 또는 정수여야 합니다 (받은 값: '$START_CORE')" >&2
  usage
fi

# ---------------- DATASET 검증 ----------------
case "$DATASET" in
  aumc|eicu|hirid|miiv) ;;
  *) echo "잘못된 DATASET: $DATASET" >&2; usage ;;
esac

# ---------------- TASK 매핑 ----------------
case "$TASK" in
  mortality24|aki|sepsis)
    TASK_TYPE=BinaryClassification
    LGBM_MODEL=LGBMClassifier
    ;;
  kidney_function|los)
    TASK_TYPE=Regression
    LGBM_MODEL=LGBMRegressor
    ;;
  *) echo "잘못된 TASK: $TASK" >&2; usage ;;
esac

TASK_NAME=$TASK

# ---------------- CPU core 자동 할당 ----------------
ALLOC_DIR=/tmp/yaib_cpu_alloc
mkdir -p "$ALLOC_DIR"
chmod 1777 "$ALLOC_DIR" 2>/dev/null || true

allocate_cpu_range() {
  local n=$1
  local total
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
    echo "ERROR: 빈 코어 $n개 못 찾음 (총 $total)" >&2
    exit 1
  fi

  echo "$found-$((found+n-1))" > "$ALLOC_DIR/$$.range"
  flock -u "$fd"
  echo "$found"
}

cleanup_alloc() {
  rm -f "$ALLOC_DIR/$$.range"
}
trap cleanup_alloc EXIT INT TERM

if [ "$START_CORE" = "auto" ]; then
  START_CORE=$(allocate_cpu_range "$NUM_THREADS")
fi
END_CORE=$((START_CORE + NUM_THREADS - 1))

# ---------------- env / cwd ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yaib

# 라이브러리 thread 수 제한
export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
export POLARS_MAX_THREADS=$NUM_THREADS
export RAYON_NUM_THREADS=$NUM_THREADS
export NUMBA_NUM_THREADS=$NUM_THREADS

mkdir -p "$LOG_DIR"
cd "$YAIB_ROOT"

if [ ! -d "$COHORTS_DATA/$TASK/$DATASET" ]; then
  echo "ERROR: 코호트 폴더 없음: $COHORTS_DATA/$TASK/$DATASET" >&2
  exit 1
fi

# ---------------- runner ----------------
run_one() {
  local model=$1
  echo
  echo "============================================================"
  echo " dataset=$DATASET  task=$TASK_NAME ($TASK_TYPE)  model=$model"
  echo " gpu=$GPU_ID  threads=$NUM_THREADS  cores=${START_CORE}-${END_CORE}"
  echo " logs -> $LOG_DIR/$DATASET/$TASK_NAME/$model/"
  echo "============================================================"
  local t0=$(date +%s)

  local RUNNER=""
  if command -v taskset >/dev/null 2>&1; then
    RUNNER="taskset -c ${START_CORE}-${END_CORE}"
  fi

  if [ "$model" = "GRU" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    $RUNNER icu-benchmarks train \
      -d "$COHORTS_DATA/$TASK/$DATASET" \
      -n "$DATASET" \
      -t "$TASK_TYPE" \
      -tn "$TASK_NAME" \
      -m GRU \
      --tune -gc -lc \
      -s "$SEED" \
      -l "$LOG_DIR/"
  else
    # LGBM은 CPU 학습 — Lightning이 GPU multi-device로 spawn하지 않도록 --cpu + GPU 숨김
    CUDA_VISIBLE_DEVICES="" \
    $RUNNER icu-benchmarks train \
      -d "$COHORTS_DATA/$TASK/$DATASET" \
      -n "$DATASET" \
      -t "$TASK_TYPE" \
      -tn "$TASK_NAME" \
      -m "$model" \
      -hp "${model}.n_jobs=$NUM_THREADS" \
      --cpu \
      --tune -gc -lc \
      -s "$SEED" \
      -l "$LOG_DIR/"
  fi

  local t1=$(date +%s)
  echo "[done] $model elapsed=$((t1 - t0))s"
}

case "${MODEL,,}" in
  lgbm|lgbmclassifier|lgbmregressor)
    run_one "$LGBM_MODEL"
    ;;
  gru)
    run_one GRU
    ;;
  both|all)
    run_one "$LGBM_MODEL"
    run_one GRU
    ;;
  *)
    echo "Unknown MODEL: $MODEL (사용 가능: lgbm | gru | both)" >&2
    exit 1
    ;;
esac
