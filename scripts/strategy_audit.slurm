#!/bin/bash
#SBATCH --job-name=strategy_audit
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Offline persuasion-strategy audit: serve the judge (Qwen3.5-35B, TP=2, needs an 80 GB card), then
# run the 42-way strategy classifier (rl.strategy_audit.run) over a finished RL run's dumped rollouts.
# This job does not log to wandb; push the per-step distribution afterwards with
# scripts/strategy_audit_to_wandb.sh.
#
#   sbatch --export=ALL,RUN=sender_qwen3-4B_stubborn_j35B_fakepen0,STEPS=1,N_GAMES=4 scripts/strategy_audit.slurm  # smoke
#   sbatch --export=ALL,RUN=sender_qwen3-4B_stubborn_j35B_fakepen0,N_GAMES=64 scripts/strategy_audit.slurm         # pilot (all steps)
#
# Env: RUN (required), STEPS (default all dumped), N_GAMES (64), MAX_WORKERS (256),
#      RETRY (unset -> RL_MONITOR_JUDGE_RETRY, else 2; the knob the online monitors read, so the
#      offline audit and training share one retry budget),
#      JUDGE_MODEL (qwen3.5-35B) / JUDGE_PATH / TP_SIZE / SGLANG_EXTRA, RL_VENV, RESULTS_ROOT,
#      MODELS_DIR (<repo>/models; the judge default is $MODELS_DIR/Qwen3.5-35B-A3B),
#      CACHE_DIR (<repo>/.cache).
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs

: "${RUN:?set RUN=<run dir under experiments/results/rl>}"
RL_VENV="${RL_VENV:-$REPO/.venv}"; PY="$RL_VENV/bin/python"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/experiments/results/rl}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen3.5-35B}"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
case "$JUDGE_MODEL" in
  qwen3.5-35B)  _MODEL_DEFAULT="$MODELS_DIR/Qwen3.5-35B-A3B"; _TP_DEFAULT=2 ;;
  *) _MODEL_DEFAULT=""; _TP_DEFAULT=2 ;;
esac
JUDGE_PATH="${JUDGE_PATH:-$_MODEL_DEFAULT}"
TP_SIZE="${TP_SIZE:-$_TP_DEFAULT}"
[[ -n "$JUDGE_PATH" ]] || { echo "ERROR: unknown JUDGE_MODEL=$JUDGE_MODEL; pass JUDGE_PATH explicitly."; exit 1; }
PORT="${PORT:-$(( 20000 + (SLURM_JOB_ID % 20000) ))}"

module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# JIT toolchain for SGLang/flashinfer (qwen3_5 mamba kernel); same as scripts/rl_serve_judge.slurm.
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

echo "[serve] $JUDGE_MODEL <- $JUDGE_PATH (TP=$TP_SIZE) on $(hostname):$PORT"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= \
  "$PY" -m sglang.launch_server \
    --model-path "$JUDGE_PATH" --served-model-name "$JUDGE_MODEL" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP_SIZE" ${SGLANG_EXTRA:-} &
SGPID=$!
trap 'kill "$SGPID" 2>/dev/null || true' EXIT

UP=0
for i in $(seq 1 180); do
  if curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] UP (attempt $i) on $(hostname):$PORT"; UP=1; break
  fi
  kill -0 "$SGPID" 2>/dev/null || { echo "[serve] died"; exit 1; }
  sleep 5
done
[[ "$UP" == 1 ]] || { echo "[serve] did not come up"; exit 1; }

# --- run the audit driver against the local judge ---
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$PORT" RECEIVER_MODEL_ID="$JUDGE_MODEL"
export RESULTS_ROOT
echo "[audit] RUN=$RUN STEPS=${STEPS:-<all>} N_GAMES=${N_GAMES:-64} MAX_WORKERS=${MAX_WORKERS:-256} RESULTS_ROOT=$RESULTS_ROOT"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= \
  "$PY" -m rl.strategy_audit.run \
    --run "$RUN" \
    ${STEPS:+--steps "$STEPS"} \
    --n-games "${N_GAMES:-64}" \
    --max-workers "${MAX_WORKERS:-256}" \
    ${RETRY:+--retry "$RETRY"}
echo "[audit] done -> $RESULTS_ROOT/$RUN/strategy_audit/"
