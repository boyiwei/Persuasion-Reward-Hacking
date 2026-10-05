#!/bin/bash
#SBATCH --job-name=serve35b_xnode
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Serve Qwen3.5-35B (receiver + audit judge) bound to 0.0.0.0, so a driver running outside this job
# can reach it: an API sender's driver (e.g. gpt-5.4-mini via the API gateway in
# datasets/old_bailey/sft/collect_rollouts.sh) or scripts/hosted_juror_login_driver.sh. Writes
# "<node>:<port>" to ENDPOINT_FILE once ready, then sends a short generation every KEEPALIVE_SEC
# until the server exits. Cancel it when the driver ends.
# Size --time (default 12h) to the whole driver grid (PROMPTS x RECEIVERS cells, each 100 val games
# x 3 rounds of a slow API sender plus a 100-game audit); if it walls first, the server dies mid-run.
#
#   sbatch --export=ALL,RECEIVER=qwen3.5-35B,ENDPOINT_FILE=/abs/path/recv_endpoint.txt,SGLANG_VENV=/.../.venv scripts/serve_receiver_xnode.slurm
# Weights come from RECEIVER_MODEL_PATH, default $MODELS_DIR/Qwen3.5-35B-A3B (MODELS_DIR defaults to
# <repo>/models). Caches go under CACHE_DIR (default <repo>/.cache).
# RECEIVER is the served-model name and must be a locally served juror (the guard rejects hosted ones).
# No `set -e`: the readiness poll's `got=$(curl|python)` fails until the server is up.
set -uo pipefail
RECEIVER="${RECEIVER:-qwen3.5-35B}"
RTP="${RECEIVER_TP:-2}"
KEEPALIVE_SEC="${KEEPALIVE_SEC:-60}"    # ping interval in seconds
KEEPALIVE_TOK="${KEEPALIVE_TOK:-32}"   # tokens generated per ping
: "${ENDPOINT_FILE:?set ENDPOINT_FILE (a path on a shared filesystem that the driver can read)}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
TC="${TC:-$MODELS_DIR}"
RPATH="${RECEIVER_MODEL_PATH:-$TC/Qwen3.5-35B-A3B}"
[ -d "$RPATH" ] || { echo "[serve] ERROR: model path missing: $RPATH"; exit 1; }
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"
[[ -x "$SGLANG_PY" ]] || { echo "[serve] ERROR: venv python not found: $SGLANG_PY"; exit 1; }

# --- juror transport guard (config/receiver/model/, evaluation/receiver_models.py) ---
# RECEIVER is the --served-model-name and the id the driver's audits ask for, so a hosted juror's id
# would serve Qwen3.5-35B weights under that name and mislabel every artifact. An id no yaml names
# prints "unknown" and exits 3: a locally served model, via RECEIVER_MODEL_PATH above. Anything else
# means the interpreter or module is broken.
_RECV_TRANSPORT="$("$SGLANG_PY" -m evaluation.receiver_models --receiver "$RECEIVER" --print-transport)" && _RECV_RC=0 || _RECV_RC=$?
case "${_RECV_TRANSPORT}:${_RECV_RC}" in
  sglang:0|unknown:3) ;;
  gateway:0)
    echo "ERROR: RECEIVER=$RECEIVER is a hosted juror (API gateway), and this job only serves local jurors. Play it from a machine with API access: scripts/hosted_juror_login_driver.sh (sender served by scripts/serve_sender_xnode.slurm, judge by scripts/serve_receiver_xnode.slurm)."
    exit 1 ;;
  *)
    echo "[serve] ERROR: cannot resolve the juror transport of RECEIVER='$RECEIVER': '$SGLANG_PY -m evaluation.receiver_models' printed '${_RECV_TRANSPORT}' and exited $_RECV_RC (is SGLANG_VENV correct?)."
    exit 1 ;;
esac

# --- caches + CUDA JIT toolchain (same as the sweep) ---
module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
_CUDA_SYS="${CUDA_SYS:-/usr/local/cuda-13.0}"
_CUDA_INC="$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13/include"
export CUDA_HOME="${CUDA_HOME:-$CACHE_DIR/cuda13-merged}"
mkdir -p "$CUDA_HOME"
ln -sfn "$_CUDA_SYS/bin" "$CUDA_HOME/bin"; ln -sfn "$_CUDA_INC" "$CUDA_HOME/include"; ln -sfn "$_CUDA_SYS/lib64" "$CUDA_HOME/lib64"
export PATH="$SGLANG_VENV/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$_CUDA_INC:${CPATH:-}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$CACHE_DIR/tvm-ffi}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_DIR/flashinfer-base}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

NODE="$(hostname)"
PORT=$(( 30000 + SLURM_JOB_ID % 20000 ))
SGLOG="logs/sglang_xnode_${SLURM_JOB_ID}.log"
echo "[serve] launching $RECEIVER on ${NODE}:${PORT} (TP=$RTP, bind 0.0.0.0)"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= \
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((RTP-1)))" \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$RPATH" --served-model-name "$RECEIVER" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$RTP" --reasoning-parser qwen3 \
    ${SGLANG_EXTRA:-} > "$SGLOG" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true; rm -f "$ENDPOINT_FILE"' EXIT

for _i in $(seq 1 300); do
  kill -0 "$SPID" 2>/dev/null || { echo "[serve] ERROR: server died. Tail:"; tail -n 80 "$SGLOG"; exit 1; }
  got="$(curl --noproxy '*' -s "http://127.0.0.1:${PORT}/v1/models" \
          | "$SGLANG_PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  [ "$got" = "$RECEIVER" ] && break
  sleep 5
done
[ "$got" = "$RECEIVER" ] || { echo "[serve] ERROR: not ready. Tail:"; tail -n 120 "$SGLOG"; exit 1; }

# `<node>:<port>` is the drivers' RECEIVER_ENDPOINT format, so
# `RECEIVER_ENDPOINT=$(cat "$ENDPOINT_FILE")` works as is.
printf '%s:%s\n' "$NODE" "$PORT" > "$ENDPOINT_FILE"
echo "[serve] READY ${RECEIVER} on ${NODE}:${PORT} -> wrote $ENDPOINT_FILE"

while kill -0 "$SPID" 2>/dev/null; do
  curl --noproxy '*' -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${RECEIVER}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":${KEEPALIVE_TOK}}" \
    >/dev/null 2>&1 || true
  sleep "$KEEPALIVE_SEC"
done
