#!/bin/bash
#SBATCH --job-name=sft_selfdistill
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Self-distillation SFT-rollout generation (stages 1-2 of the SFT-then-RL curriculum, teacher = a
# self-hosted Qwen instead of gpt-5.4-mini). One GPU job co-serves the sender (Qwen
# teacher/checkpoint, TP=1) and the Qwen3.5-35B receiver/judge (TP=2) on disjoint GPUs, then runs
# collect_rollouts.sh with SENDER_API=sglang against the local endpoints; no external network.
#
# The driver plays the passes and freezes the per-turn audit sidecars (see its header); with
# BUILD_SFT=1 (default) this job then builds the SFT parquets via build_sft_dataset.py.
#
# Submit one teacher per job from the repo root (the co-served 35B needs an 80 GB card). The
# self-distill SFT inits are collected with:
#   sbatch --export=ALL,SENDER=qwen3-8B-base,PASSES=10,STEER_K=0,BUILD_SFT=0 \
#       datasets/old_bailey/sft/collect_selfdistill.slurm
#   sbatch --export=ALL,SENDER=qwen3-4B-instruct-base,PASSES=10,STEER_K=0,BUILD_SFT=0 \
#       datasets/old_bailey/sft/collect_selfdistill.slurm
#   # any served checkpoint (not a resolve_model key): pass its HF dir
#   sbatch --export=ALL,SENDER=my-ckpt,SENDER_MODEL_PATH=/abs/hf_dir datasets/old_bailey/sft/collect_selfdistill.slurm
#   # smoke (2 games, 1 pass, 1 profile). Two games are too few for the clean-prefix filters (the
#   # scaffolding-leak guard alone usually rejects both at round 1) or a val holdout, so the smoke
#   # lifts the filters and takes no val split; otherwise the builder exits on an empty dataset.
#   # BUILD_EXTRA holds spaces and --export splits on commas, so export the knobs and pass a bare
#   # --export=ALL. BUILD_EXTRA is appended last, so its --max-illegal-per-turn overrides MAX_ILLEGAL:
#   export SENDER=qwen3-4B-instruct-base END_IDX=2 PASSES=1 STEER_K=0 VAL_GAMES=0
#   export BUILD_EXTRA="--allow-harness-leakage --max-illegal-per-turn 11 --skip-fabrication-filter"
#   sbatch --export=ALL datasets/old_bailey/sft/collect_selfdistill.slurm
#
# Hosted gpt-5.4-mini teacher: serve the 35B with scripts/serve_receiver_xnode.slurm and run
# datasets/old_bailey/sft/collect_rollouts.sh with SENDER_API=gateway from a machine with API access
# (see its header).
#
# Required: SENDER (a sender model key; see resolve_model below, or pass SENDER_MODEL_PATH).
# Optional: RECEIVER(qwen3.5-35B) PROFILE(stubborn) PROMPT(strategies|base) PASSES(3)
#           NUM_STEPS(3) END_IDX(all) MAX_WORKERS(16) AUDIT_WORKERS(64) FAB_WORKERS(32)
#           STEER_K(4) STEER_FROM_PASS(2) SKIP(true) FORCE_AUDIT(0)
#           SENDER_TP/RECEIVER_TP(default = resolve_model TP: 1 sender / 2 for the 35B receiver)
#           SENDER_MODEL_PATH/RECEIVER_MODEL_PATH/SGLANG_EXTRA
#           BUILD_SFT(1) OUT_DIR STUDENT_TOKENIZER PROMPTS_PARQUET MAX_ILLEGAL(0) VAL_GAMES(10)
#           BUILD_EXTRA (extra args appended to build_sft_dataset.py, e.g.
#             "--skip-fabrication-filter --select-diverse 5"; plumbing tests / filter tuning)
#           SGLANG_VENV(main .venv) INFER_PYTHON MODELS_DIR(<repo>/models) CACHE_DIR(<repo>/.cache)
set -euo pipefail

: "${SENDER:?Must set SENDER (a sender model key; see resolve_model below, or pass SENDER_MODEL_PATH)}"
RECEIVER="${RECEIVER:-qwen3.5-35B}"
PROFILE="${PROFILE:-stubborn}"            # single receiver profile; must match the GRPO DIST
PROMPT="${PROMPT:-strategies}"            # sender prompt knob: strategies|base
PASSES="${PASSES:-3}"
NUM_STEPS="${NUM_STEPS:-3}"
END_IDX="${END_IDX:-}"                    # cap after id selection; empty = all 200 SFT-train games
MAX_WORKERS="${MAX_WORKERS:-16}"
BUILD_SFT="${BUILD_SFT:-1}"              # 1 = build the SFT parquets from result+sidecars after the passes
# SENDER_TP / RECEIVER_TP are left unset so STP/RTP below default to resolve_model's per-model TP.

# REPO = submit dir: its .venv, generated data roots and sft_holdout.parquet live there, so all
# relative paths resolve against it. The fallback is three levels above this script.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"; mkdir -p logs
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"

# --- model key -> checkpoint path + tensor-parallel size (subset of oldbailey_rl_eval.slurm) ---
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
TC="${TC:-$MODELS_DIR}"
resolve_model() {   # $1 = model key -> sets RM_PATH, RM_TP
  case "$1" in
    qwen3-4B-instruct-base) RM_PATH="$TC/Qwen3-4B-Instruct-2507";     RM_TP=1 ;;
    qwen3-8B-base)          RM_PATH="$MODELS_DIR/Qwen3-8B";          RM_TP=1 ;;
    qwen3-14B-base)         RM_PATH="$TC/Qwen3-14B";                  RM_TP=1 ;;
    qwen3.5-35B)            RM_PATH="$TC/Qwen3.5-35B-A3B";            RM_TP=2 ;;
    *)                      RM_PATH="";                               RM_TP=1 ;;  # unknown -> require SENDER_MODEL_PATH
  esac
}

resolve_model "$SENDER";   SPATH="${SENDER_MODEL_PATH:-$RM_PATH}";   STP="${SENDER_TP:-$RM_TP}"
resolve_model "$RECEIVER"; RPATH="${RECEIVER_MODEL_PATH:-$RM_PATH}"; RTP="${RECEIVER_TP:-$RM_TP}"
[ -n "$SPATH" ] || { echo "[serve] ERROR: unknown SENDER '$SENDER'; pass SENDER_MODEL_PATH=/abs/hf_dir."; exit 1; }
[ -n "$RPATH" ] || { echo "[serve] ERROR: unknown RECEIVER '$RECEIVER'; pass RECEIVER_MODEL_PATH."; exit 1; }
[ -d "$SPATH" ] || { echo "[serve] ERROR: SENDER path missing: $SPATH."; exit 1; }
[ -d "$RPATH" ] || { echo "[serve] ERROR: RECEIVER path missing: $RPATH."; exit 1; }

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
NEED=$(( STP + RTP ))
if (( NEED > NGPU )); then
  echo "[serve] ERROR: need $NEED GPU(s) (sender TP=$STP + receiver TP=$RTP) but only $NGPU visible. Resubmit with --gres=gpu:${NEED}."
  exit 1
fi
SGPUS="$(seq -s, 0 $((STP-1)))"
RGPUS="$(seq -s, "$STP" $((STP+RTP-1)))"

# --- network / caches; JIT toolchain for SGLang/flashinfer kernels ---
module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_DIR/wandb}"
export WANDB_MODE=offline
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

# --- env checks ---
[[ -x "$SGLANG_PY" ]] || { echo "[env] ERROR: venv python not found at '$SGLANG_PY' (set SGLANG_VENV)."; exit 1; }
"$SGLANG_PY" -c "import sglang" >/dev/null 2>&1 || { echo "[env] ERROR: '$SGLANG_PY' cannot import sglang."; exit 1; }
INFER_PYTHON="${INFER_PYTHON:-$SGLANG_PY}"; unset PYTHONPATH || true
"$INFER_PYTHON" -c "import openai, agents, sft.sft_dataset, rl.strategy_audit.run, rl.receiver_client" >/dev/null 2>&1 \
  || { echo "[env] ERROR: INFER_PYTHON lacks openai/agents/sft/rl.* (verl importable?)."; exit 1; }

# --- boot both SGLang servers (sender on GPU0, receiver/judge=35B on the next GPUs) ---
SPORT=$(( 20000 + 2 * (SLURM_JOB_ID % 17000) ))
RPORT=$(( SPORT + 1 ))
SSGLOG="logs/sglang_sender_${SLURM_JOB_ID}.log"; RSGLOG="logs/sglang_recv_${SLURM_JOB_ID}.log"
echo "[serve] sender   $SENDER <- $SPATH (TP=$STP, GPU $SGPUS) on :$SPORT"
echo "[serve] receiver $RECEIVER <- $RPATH (TP=$RTP, GPU $RGPUS) on :$RPORT  (receiver doubles as judge)"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES="$SGPUS" \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$SPATH" --served-model-name "$SENDER" \
    --host 127.0.0.1 --port "$SPORT" --tp-size "$STP" --reasoning-parser qwen3 \
    ${SGLANG_EXTRA:-} > "$SSGLOG" 2>&1 &
SPID=$!
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES="$RGPUS" \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$RPATH" --served-model-name "$RECEIVER" \
    --host 127.0.0.1 --port "$RPORT" --tp-size "$RTP" --reasoning-parser qwen3 \
    ${SGLANG_EXTRA:-} > "$RSGLOG" 2>&1 &
RPID=$!
cleanup() { kill "$SPID" "$RPID" 2>/dev/null || true; }
trap cleanup EXIT

wait_ready() {  # $1=port $2=pid $3=log $4=name $5=expected served-model-name
  for _i in $(seq 1 240); do
    if ! kill -0 "$2" 2>/dev/null; then echo "[serve] ERROR: $4 process died. Tail of $3:"; tail -n 80 "$3"; return 1; fi
    got="$(curl --noproxy '*' -s "http://127.0.0.1:$1/v1/models" 2>/dev/null \
            | "$INFER_PYTHON" -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
    if [ "$got" = "$5" ]; then echo "[serve] $4 UP ($got on :$1)"; return 0; fi
    if [ -n "$got" ]; then
      echo "[serve] ERROR: :$1 serves '$got' but $4 expected '$5' — PORT COLLISION. Tail of $3:"; tail -n 40 "$3"; return 1
    fi
    sleep 5
  done
  echo "[serve] ERROR: $4 not ready. Tail of $3:"; tail -n 120 "$3"; return 1
}
wait_ready "$SPORT" "$SPID" "$SSGLOG" "sender"   "$SENDER"   || exit 1
wait_ready "$RPORT" "$RPID" "$RSGLOG" "receiver" "$RECEIVER" || exit 1

# --- run the SFT-rollout driver against the two local endpoints. Export the knobs: in an inline
#     `VAR=val ... cmd` prefix, a `${END_IDX:+...}` word after literal assignments is parsed as the
#     command. STEER_K, STEER_FROM_PASS, AUDIT_WORKERS, FAB_WORKERS, SKIP, FORCE_AUDIT come in via
#     the inherited --export=ALL job env. ---
export PYTHON="$INFER_PYTHON"
export SENDER_API=sglang
export SENDER_ENDPOINT="127.0.0.1:${SPORT}"
export RECEIVER_ENDPOINT="127.0.0.1:${RPORT}"
export SENDER RECEIVER PROFILE PROMPT PASSES NUM_STEPS MAX_WORKERS END_IDX
echo "=== [selfdistill] driver: teacher=$SENDER receiver=$RECEIVER profile=$PROFILE prompt=$PROMPT passes=$PASSES games=${END_IDX:-all} ==="
bash "$REPO/datasets/old_bailey/sft/collect_rollouts.sh" || { echo "[selfdistill] ERROR: driver failed."; exit 1; }

# --- optionally build the SFT parquets from the frozen result+sidecars. The student tokenizer
#     defaults to the teacher path (self-distill: teacher == student; override when they differ). ---
if [ "$BUILD_SFT" = 1 ]; then
  case "$PROMPT" in
    strategies) _ROOT="rl_sftsplit_strategies" ;;
    base)       _ROOT="rl_sftsplit" ;;
    *) echo "[build] ERROR: unknown PROMPT '$PROMPT'"; exit 1 ;;
  esac
  _DIST="${PROFILE/neutral/bayesian}"
  # sft_holdout.parquet lives under this checkout's data root; override with PROMPTS_PARQUET.
  PROMPTS_PARQUET="${PROMPTS_PARQUET:-$REPO/datasets/old_bailey/_generated/${_ROOT}/${_DIST}/sft_holdout.parquet}"
  STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-$SPATH}"
  SENDER_SLUG="${SENDER//\//-}"
  OUT_DIR="${OUT_DIR:-datasets/old_bailey/_generated/sft/${PROMPT}_${PROFILE}_${SENDER_SLUG}}"
  # List the pass result files exactly: a `pass*.json` glob would also match the per-pass
  # sidecars, which the builder would misread as results.
  RES_BASE="experiments/results/old-bailey/${SENDER_SLUG}/sft_rollouts/${PROFILE}_sfttrain_${PROMPT}"
  RES_FILES=(); for (( p=1; p<=PASSES; p++ )); do RES_FILES+=("${RES_BASE}_pass${p}.json"); done
  if [ ! -f "$PROMPTS_PARQUET" ]; then
    echo "[build] WARN: prompts-parquet missing ($PROMPTS_PARQUET) — SKIPPING SFT build. Rollouts+audits are done; run build_sft_dataset.py manually."
  else
    echo "[build] SFT dataset -> $OUT_DIR  (prompts=$PROMPTS_PARQUET, tokenizer=$STUDENT_TOKENIZER, extra='${BUILD_EXTRA:-}')"
    # shellcheck disable=SC2086  # BUILD_EXTRA is intentionally word-split into extra flags
    "$INFER_PYTHON" "$REPO/datasets/old_bailey/sft/build_sft_dataset.py" --results "${RES_FILES[@]}" \
      --prompts-parquet "$PROMPTS_PARQUET" \
      --tokenizer "$STUDENT_TOKENIZER" \
      --max-illegal-per-turn "${MAX_ILLEGAL:-0}" \
      --val-games "${VAL_GAMES:-10}" \
      --out-dir "$OUT_DIR" ${BUILD_EXTRA:-} || { echo "[build] ERROR: SFT build failed."; exit 1; }
    echo "[build] done -> $OUT_DIR/{sft_train,sft_val}.parquet + sft_build_stats.json"
  fi
fi

echo "[done] self-distill SFT-rollout generation complete for teacher=$SENDER (profile=$PROFILE, prompt=$PROMPT)."
