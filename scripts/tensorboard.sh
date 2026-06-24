#!/bin/bash
# TensorBoard launcher for YAIB experiment logs
#
# Usage:
#   ./tensorboard.sh -l LOGDIR [-p PORT] [-b BIND_HOST]
#
#   -l  LOGDIR     TensorBoard 로그 디렉토리 (필수)
#                  event 파일을 재귀로 찾으므로 가리키는 레벨에 따라 범위가 결정됨:
#                    .../<run-timestamp>            한 run 전체 (rep×fold)
#                    .../<model>                    모델의 모든 run 비교
#                    .../<task>                     모델 간 비교
#   -p  PORT       포트                      (default: 6006)
#   -b  BIND_HOST  바인드 호스트             (default: 0.0.0.0 = 외부 접속 허용)
#
# Examples:
#   ./tensorboard.sh -l /team/.../logs/hirid/mortality24/mTAND/2026-06-23T19-22-20
#   ./tensorboard.sh -l /team/.../logs/hirid/mortality24/mTAND -p 6007
#
# 접속 (로컬 PC에서 SSH 포워딩):
#   ssh -L PORT:localhost:PORT <user>@<host>   →   http://localhost:PORT

set -euo pipefail

# ======================== 기본값 ========================
LOG_BASE=/team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs
CONDA_ENV=yaib
PORT=6006
BIND_HOST=0.0.0.0
LOGDIR=""
# ========================================================

usage() {
  cat <<USAGE >&2
Usage: $0 -l LOGDIR [-p PORT] [-b BIND_HOST]

  -l  LOGDIR     TensorBoard 로그 디렉토리 (필수, 절대경로 또는 $LOG_BASE 기준 상대경로)
  -p  PORT       포트            (default: 6006)
  -b  BIND_HOST  바인드 호스트   (default: 0.0.0.0)
USAGE
  exit 1
}

while getopts "l:p:b:h" opt; do
  case "$opt" in
    l) LOGDIR=$OPTARG ;;
    p) PORT=$OPTARG ;;
    b) BIND_HOST=$OPTARG ;;
    h|*) usage ;;
  esac
done

[ -z "$LOGDIR" ] && { echo "ERROR: -l LOGDIR 는 필수입니다" >&2; usage; }

# 포트 검증
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "ERROR: -p PORT 는 1~65535 정수여야 합니다 (받은 값: '$PORT')" >&2
  usage
fi

# 상대경로면 LOG_BASE 기준으로 해석 (편의)
if [ ! -e "$LOGDIR" ] && [ -e "$LOG_BASE/$LOGDIR" ]; then
  LOGDIR="$LOG_BASE/$LOGDIR"
fi
if [ ! -d "$LOGDIR" ]; then
  echo "ERROR: 로그 디렉토리 없음: $LOGDIR" >&2
  exit 1
fi
LOGDIR="$(cd "$LOGDIR" && pwd -P)"   # 절대경로로 정규화

# 포트 점유 검사
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "ERROR: 포트 $PORT 가 이미 사용 중입니다. 다른 -p 포트를 쓰거나 기존 프로세스를 종료하세요:" >&2
  echo "       pkill -f \"tensorboard.*--port $PORT\"" >&2
  exit 1
fi

# conda 환경 활성화 (영구 fix: setuptools<81 가 yaib 환경에 설치돼 있어야 함)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

HOST="$(hostname)"
# primary IP (docker/bridge 인터페이스 172.1[78].* 등은 제외하고 첫 실IP)
HOST_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^172\.1[78]\.' | head -n1)"
[ -z "$HOST_IP" ] && HOST_IP="$HOST"
echo "============================================================"
echo " TensorBoard"
echo "   logdir : $LOGDIR"
echo "   host   : $BIND_HOST   port: $PORT   ($HOST / $HOST_IP)"
echo "------------------------------------------------------------"
echo " 접속:  ssh -L $PORT:localhost:$PORT $USER@$HOST_IP"
echo "        → http://localhost:$PORT"
echo "   (같은 네트워크 직접 접속:  http://$HOST_IP:$PORT )"
echo "============================================================"

# 데이터 로딩만 필요하므로 thread 1개로 제한 (실행 중 실험 영향 최소화)
exec env OMP_NUM_THREADS=1 \
  tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$BIND_HOST" --load_fast=false
