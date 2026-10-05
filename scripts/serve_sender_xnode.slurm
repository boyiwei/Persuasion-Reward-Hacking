#!/bin/bash
#SBATCH --job-name=serve_sender_xnode
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Serve a sequence of sender checkpoints bound to 0.0.0.0, so a driver running outside this job
# (scripts/hosted_juror_login_driver.sh) can play games against them. Mirror of
# scripts/serve_receiver_xnode.slurm: here the sender is local and the receiver is hosted
# (DeepSeek-V4-Flash via the API gateway), which the driver calls.
#
# One checkpoint fits at a time, so for each row of CELLS_FILE:
#   1. serve it and wait until /v1/models reports its slug
#   2. write "<node>:<port>:<slug>" to ENDPOINT_FILE for the driver
#   3. wait for the driver to write that slug into $ENDPOINT_FILE.done
#   4. kill the server and move to the next row
# During the wait it sends a short generation to the server every KEEPALIVE_SEC.
#
#   sbatch --export=ALL,CELLS_FILE=/abs/cells.tsv,ENDPOINT_FILE=/abs/sender_endpoint.txt \
#          scripts/serve_sender_xnode.slurm
#
# Caches go under CACHE_DIR (default <repo>/.cache).
# No `set -e`: the readiness poll's `got=$(curl|python)` fails until the server is up.
set -uo pipefail
: "${CELLS_FILE:?set CELLS_FILE (tsv: cell_key<TAB>slug<TAB>ckpt_path<TAB>stem_kind)}"
: "${ENDPOINT_FILE:?set ENDPOINT_FILE (a path on a shared filesystem that the driver can read)}"
STP="${SENDER_TP:-1}"
KEEPALIVE_SEC="${KEEPALIVE_SEC:-60}"
KEEPALIVE_TOK="${KEEPALIVE_TOK:-32}"
CELL_TIMEOUT="${CELL_TIMEOUT:-7200}"   # give up on a cell after this many seconds of driver silence

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"
[[ -x "$SGLANG_PY" ]] || { echo "[serve] ERROR: venv python not found: $SGLANG_PY"; exit 1; }
[[ -f "$CELLS_FILE" ]] || { echo "[serve] ERROR: CELLS_FILE not found: $CELLS_FILE"; exit 1; }

# --- optional job-level claim for duplicate submissions ---
# Set CLAIM_NAME when the same serve job is submitted more than once; otherwise both copies write the
# same ENDPOINT_FILE and the driver ping-pongs between two live endpoints. Unset = no claim.
# This must run before `trap cleanup EXIT`: cleanup() removes ENDPOINT_FILE, so a job exiting 3 after
# the trap would delete the claim holder's endpoint file and strand its driver.
CLAIM_NAME="${CLAIM_NAME:-}"
CLAIM_DIR=""
release_claim() {
  [[ -n "$CLAIM_DIR" && "$(cat "$CLAIM_DIR/owner" 2>/dev/null)" == "${SLURM_JOB_ID:-0}" ]] && rm -rf "$CLAIM_DIR"
  return 0
}
if [[ -n "$CLAIM_NAME" ]]; then
  CLAIM_ROOT="$REPO/logs/claims"; CLAIM="$CLAIM_ROOT/$CLAIM_NAME"
  mkdir -p "$CLAIM_ROOT"
  if ! mkdir "$CLAIM" 2>/dev/null; then
    owner="$(tr -dc '0-9' < "$CLAIM/owner" 2>/dev/null || true)"
    # mkdir and the owner write are separate steps, so a job arriving in between reads an empty
    # owner. Treating that as stale would take over a live claim, so an owner still empty after one
    # re-read means another job is mid-claim: stand down. A claim abandoned between the two steps
    # blocks later jobs until its directory under logs/claims/ is removed by hand.
    if [[ -z "$owner" ]]; then
      sleep 2
      owner="$(tr -dc '0-9' < "$CLAIM/owner" 2>/dev/null || true)"
      if [[ -z "$owner" ]]; then
        echo "[claim] $CLAIM exists but carries no owner yet -- another job is mid-claim."
        echo "[claim] Exiting 3 WITHOUT touching $ENDPOINT_FILE."
        exit 3
      fi
    fi
    if squeue -h -j "$owner" -o "%t" 2>/dev/null | grep -q .; then
      echo "[claim] $CLAIM is held by job $owner (still queued/running): this job stands down."
      echo "[claim] Exiting 3 WITHOUT touching $ENDPOINT_FILE."
      exit 3
    fi
    echo "[claim] stale claim (owner '${owner:-?}' not in squeue) -- taking over"
    rm -rf "$CLAIM"; mkdir "$CLAIM" || { echo "[claim] ERROR: cannot create $CLAIM"; exit 1; }
  fi
  echo "${SLURM_JOB_ID:-0}" > "$CLAIM/owner"; CLAIM_DIR="$CLAIM"
  echo "[claim] job ${SLURM_JOB_ID:-0} holds $CLAIM"
fi

# --- caches + CUDA JIT toolchain (same as serve_receiver_xnode.slurm) ---
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
DONE_FILE="${ENDPOINT_FILE}.done"
SPID=""
# release_claim is a no-op when CLAIM_NAME is unset.
cleanup() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; rm -f "$ENDPOINT_FILE"; release_claim; }
trap cleanup EXIT

rm -f "$ENDPOINT_FILE" "$DONE_FILE"
echo "[serve] node=$NODE port=$PORT cells=$CELLS_FILE"

n_rows=$(grep -cve '^[[:space:]]*$' "$CELLS_FILE")
row=0
while IFS=$'\t' read -r CELL SLUG CKPT KIND; do
    [ -z "${SLUG:-}" ] && continue
    row=$((row+1))
    if [ ! -d "$CKPT" ]; then
        echo "[serve] SKIP ($row/$n_rows) $CELL: checkpoint missing: $CKPT"; continue
    fi
    echo "[serve] ==== ($row/$n_rows) $CELL -> $SLUG"
    SGLOG="logs/sglang_sender_${SLURM_JOB_ID}_${SLUG}.log"
    HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= \
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((STP-1)))" \
      "$SGLANG_PY" -m sglang.launch_server --model-path "$CKPT" --served-model-name "$SLUG" \
        --host 0.0.0.0 --port "$PORT" --tp-size "$STP" --reasoning-parser qwen3 \
        ${SGLANG_EXTRA:-} > "$SGLOG" 2>&1 &
    SPID=$!

    got=""
    for _i in $(seq 1 300); do
        kill -0 "$SPID" 2>/dev/null || break
        got="$(curl --noproxy '*' -s "http://127.0.0.1:${PORT}/v1/models" \
                | "$SGLANG_PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
        [ "$got" = "$SLUG" ] && break
        sleep 5
    done
    if [ "$got" != "$SLUG" ]; then
        echo "[serve] ERROR: $SLUG never became ready; tail:"; tail -n 60 "$SGLOG"
        kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""
        continue                     # don't sink the whole grid over one bad checkpoint
    fi

    # Hand off. The slug in the payload keeps the driver from acting on a stale endpoint.
    printf '%s:%s:%s\n' "$NODE" "$PORT" "$SLUG" > "$ENDPOINT_FILE"
    echo "[serve] READY $SLUG on ${NODE}:${PORT} (waiting for driver)"

    waited=0
    while true; do
        if [ -f "$DONE_FILE" ] && grep -qxF "$SLUG" "$DONE_FILE"; then
            echo "[serve] driver finished $SLUG after ${waited}s"; break
        fi
        kill -0 "$SPID" 2>/dev/null || { echo "[serve] ERROR: server for $SLUG died while waiting"; break; }
        [ "$waited" -ge "$CELL_TIMEOUT" ] && { echo "[serve] WARN: $CELL_TIMEOUT s with no driver ack for $SLUG; advancing"; break; }
        curl --noproxy '*' -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d "{\"model\":\"${SLUG}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":${KEEPALIVE_TOK}}" \
          >/dev/null 2>&1 || true
        sleep "$KEEPALIVE_SEC"; waited=$((waited+KEEPALIVE_SEC))
    done

    rm -f "$ENDPOINT_FILE"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""
    sleep 5                          # let the port drain before the next bind
done < "$CELLS_FILE"

echo "[serve] all $n_rows cells served; exiting"
