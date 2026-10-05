#!/bin/bash
#SBATCH --job-name=rl_serve_judge
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Serve the receiver + judge for sender RL from the repo .venv as its own job, for trainers that
# cannot co-locate it (qwen3-14B); by default scripts/rl_train_sender.slurm serves it inline.
# JUDGE_MODEL (qwen3.5-35B: bf16, TP=2); other ids need MODEL_PATH. Once up, host:port is written to
# ENDPOINT_FILE; pass that file to the trainer as JUDGE_ENDPOINT_FILE. The default checkpoint is
# MODELS_DIR/Qwen3.5-35B-A3B (MODELS_DIR defaults to <repo>/models); caches go under CACHE_DIR
# (<repo>/.cache).
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"

RL_VENV="${RL_VENV:-$REPO/.venv}"; PY="$RL_VENV/bin/python"
JUDGE_MODEL="${JUDGE_MODEL:-qwen3.5-35B}"   # qwen3.5-35B (bf16, TP=2)
case "$JUDGE_MODEL" in
  qwen3.5-35B)  _MODEL_DEFAULT="$MODELS_DIR/Qwen3.5-35B-A3B";       _TP_DEFAULT=2 ;;
  *) _MODEL_DEFAULT=""; _TP_DEFAULT=2 ;;   # custom model: pass MODEL_PATH explicitly
esac
MODEL_PATH="${MODEL_PATH:-$_MODEL_DEFAULT}"
TP_SIZE="${TP_SIZE:-$_TP_DEFAULT}"   # --gres=gpu:2 fits TP<=2; raise --gres for larger TP
[[ -n "$MODEL_PATH" ]] || { echo "ERROR: unknown JUDGE_MODEL=$JUDGE_MODEL; pass MODEL_PATH explicitly."; exit 1; }
PORT="${PORT:-$(( 20000 + (SLURM_JOB_ID % 20000) ))}"
ENDPOINT_FILE="${ENDPOINT_FILE:-$REPO/logs/judge_endpoint.txt}"  # the trainer reads host:port from here

module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # local ckpts only
# JIT toolchain for SGLang/flashinfer: the qwen3_5 mamba kernel is tvm_ffi-JIT'd at CUDA-graph
# capture and needs ninja, nvcc and full cu13 headers. Mirrors scripts/oldbailey_rl_eval.slurm.
_CUDA_SYS="${CUDA_SYS:-/usr/local/cuda-13.0}"
_CUDA_INC="$RL_VENV/lib/python3.12/site-packages/nvidia/cu13/include"
export CUDA_HOME="${CUDA_HOME:-$CACHE_DIR/cuda13-merged}"
mkdir -p "$CUDA_HOME"
ln -sfn "$_CUDA_SYS/bin" "$CUDA_HOME/bin"; ln -sfn "$_CUDA_INC" "$CUDA_HOME/include"; ln -sfn "$_CUDA_SYS/lib64" "$CUDA_HOME/lib64"
export PATH="$RL_VENV/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$_CUDA_INC:${CPATH:-}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$CACHE_DIR/tvm-ffi}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_DIR/flashinfer-base}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

echo "[serve] $JUDGE_MODEL <- $MODEL_PATH (TP=$TP_SIZE) on $(hostname):$PORT"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= \
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$JUDGE_MODEL" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP_SIZE" ${SGLANG_EXTRA:-} &
SGPID=$!
trap 'rm -f "$ENDPOINT_FILE" 2>/dev/null; kill "$SGPID" 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  if curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "$(hostname):$PORT" > "$ENDPOINT_FILE"
    echo "[serve] UP (attempt $i). published $(cat "$ENDPOINT_FILE") -> $ENDPOINT_FILE"; break
  fi
  kill -0 "$SGPID" 2>/dev/null || { echo "[serve] died"; exit 1; }
  sleep 5
done
# Keepalive: a 4-token generation every KEEPALIVE_SEC seconds. 0 disables.
KEEPALIVE_SEC="${KEEPALIVE_SEC:-60}"
if [[ "$KEEPALIVE_SEC" != 0 ]]; then
  ( while sleep "$KEEPALIVE_SEC"; do
      curl --noproxy '*' -s -m 30 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" \
        >/dev/null 2>&1 || true
    done ) &
  KAPID=$!
  trap 'kill "$KAPID" 2>/dev/null || true; rm -f "$ENDPOINT_FILE" 2>/dev/null; kill "$SGPID" 2>/dev/null || true' EXIT
fi
wait "$SGPID"
