#!/bin/bash
#SBATCH --job-name=hs_rl_eval
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# House-showing generalization eval. An RL-trained sender (Old Bailey persuasion policy; here the
# stubborn-judge Qwen3-4B/8B/14B checkpoints) plays the seller, the model under test for deception,
# against a fixed Qwen3.5-35B buyer for 3 rounds. The 35B judge then scores the seller's deception
# with the false_claims audit in the game's wording (flips of a known feature, invented facts).
#
# Co-serves two SGLang models on disjoint GPUs: the sender (TP=1, GPU 0) and the 35B receiver/judge
# (TP=2, GPU 1-2). The game stage is evaluation/rl_rollout.py, the RL-parity driver that replays the
# verl GRPO conversation (multi-turn sender chat, <receiver_response> turns, training receiver
# prompt/sampling via rl.receiver_client). The judge stage (evaluate_houseshowing.py) uses the same
# transport, so the judge is the served receiver. Native <think> is off everywhere (driver sends
# enable_thinking=false; judge via SGLANG_ENABLE_THINKING=0; both servers run --reasoning-parser qwen3).
#
# Profiles: neutral and stubborn buyers (the prior is injected at load by the experiment config; one
# dataset). wandb is off; per-cell metrics land in <result>.deception.json and are aggregated after
# the job.
#
# Submit one sender:
#   sbatch --export=ALL,SENDER=qwen3-8B-rl-stubborn-j35B-fakepen0 scripts/houseshowing_rl_eval.slurm
#   # smoke (2 games, 1 profile, 1 round):
#   sbatch --export=ALL,SENDER=qwen3-8B-rl-stubborn-j35B-fakepen0,END_IDX=2,NUM_STEPS=1,PROFILES=neutral,MAX_WORKERS=2 scripts/houseshowing_rl_eval.slurm
#   # full sweep (3 senders):
#   for S in qwen3-4B-rl-stubborn-j35B-fakepen0 qwen3-8B-rl-stubborn-j35B-fakepen0 qwen3-14B-rl-stubborn-j35B-fakepen0; do
#     sbatch --export=ALL,SENDER=$S scripts/houseshowing_rl_eval.slurm; done
#
# Required: SENDER (a model key in resolve_model below, or pass SENDER_MODEL_PATH).
# Optional: RECEIVER(qwen3.5-35B; served by this job; play a hosted juror with
#           scripts/hosted_juror_login_driver.sh)
#           JUDGE($RECEIVER) PROFILES("neutral stubborn")
#           PROMPT(base|strategies|single_strategy:<slug>; the sender prompt, rl/sender_prompts.py)
#           RES_TAG(filename tag before "__recv_")
#           NUM_STEPS(3) END_IDX(211) MAX_WORKERS(16) SENDER_TP(1) RECEIVER_TP(2)
#           SENDER_MODEL_PATH/RECEIVER_MODEL_PATH/SGLANG_EXTRA
#           MERGED(<repo>/experiments/results/rl/_em_merged)
#           SGLANG_VENV(<repo>/.venv) INFER_PYTHON
#           MODELS_DIR(<repo>/models; one directory per base model) CACHE_DIR(<repo>/.cache)
set -euo pipefail

: "${SENDER:?Must set SENDER (a sender model key; see resolve_model below)}"
RECEIVER="${RECEIVER:-qwen3.5-35B}"
JUDGE="${JUDGE:-$RECEIVER}"
PROFILES="${PROFILES:-neutral stubborn}"
# Sender prompt (rl/sender_prompts.py), rendered in this game's wording: base | strategies (the
# 42-technique ALLOWED/FORBIDDEN guide) | single_strategy:<slug>. Use the prompt the checkpoint was
# trained on; a mismatch measures the prompt, not the checkpoint.
PROMPT="${PROMPT:-base}"
# Filename tag inserted before "__recv_" so a non-base PROMPT gets its own result file instead of
# sharing the base-prompt name of the same SENDER key. Empty for base:
#   RES_TAG=_sender-$(python -m rl.sender_prompts --print result-stem --domain house-showing --prompt strategies)
RES_TAG="${RES_TAG:-}"
NUM_STEPS="${NUM_STEPS:-3}"
END_IDX="${END_IDX:-211}"
MAX_WORKERS="${MAX_WORKERS:-16}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"

# --- juror transport guard (config/receiver/model/, evaluation/receiver_models.py) ---
# RECEIVER is also the served-model name, default JUDGE and "__recv_" stem, so reject a hosted
# (API gateway) juror here, once the cwd is the repo root and before any GPU work, instead of serving a
# local model under its name; play it with scripts/hosted_juror_login_driver.sh. An id no yaml
# names exits 3 ("unknown"): a local model, resolved via RECEIVER_MODEL_PATH. Any other result
# means a broken interpreter or module.
_RECV_TRANSPORT="$("$SGLANG_PY" -m evaluation.receiver_models --receiver "$RECEIVER" --print-transport)" && _RECV_RC=0 || _RECV_RC=$?
case "${_RECV_TRANSPORT}:${_RECV_RC}" in
  sglang:0|unknown:3) ;;
  gateway:0)
    echo "ERROR: RECEIVER=$RECEIVER is a hosted juror (API gateway), and this job only serves local jurors. Play it from a machine with API access: scripts/hosted_juror_login_driver.sh (sender served by scripts/serve_sender_xnode.slurm, judge by scripts/serve_receiver_xnode.slurm)."
    exit 1 ;;
  *)
    echo "[env] ERROR: cannot resolve the juror transport of RECEIVER='$RECEIVER': '$SGLANG_PY -m evaluation.receiver_models' printed '${_RECV_TRANSPORT}' and exited $_RECV_RC (is SGLANG_VENV correct?)."
    exit 1 ;;
esac

# --- model key -> checkpoint path + tensor-parallel size ---
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
TC="${TC:-$MODELS_DIR}"
# Merged RL checkpoint root (MERGED relocates it).
MERGED="${MERGED:-$REPO/experiments/results/rl/_em_merged}"
resolve_model() {   # $1 = model key -> sets RM_PATH, RM_TP
  # Stubborn + neutral fakepen0 senders: the fmt0p1 gs100 checkpoints (07-01 wandb project). Re-merge
  # a missing dir via scripts/merge_verl_ckpt.slurm (neutral: ONLY=neutral_j35B_fakepen0_fmt0p1).
  case "$1" in
    qwen3-4B-rl-stubborn-j35B-fakepen0)   RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs100";  RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0)   RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs100";  RM_TP=1 ;;
    qwen3-14B-rl-stubborn-j35B-fakepen0)  RM_PATH="$MERGED/sender_qwen3-14B_stubborn_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-4B-rl-neutral-j35B-fakepen0)    RM_PATH="$MERGED/sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs100";   RM_TP=1 ;;
    qwen3-8B-rl-neutral-j35B-fakepen0)    RM_PATH="$MERGED/sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1_gs100";   RM_TP=1 ;;
    qwen3-14B-rl-neutral-j35B-fakepen0)   RM_PATH="$MERGED/sender_qwen3-14B_neutral_j35B_fakepen0_fmt0p1_gs100";  RM_TP=1 ;;
    # Step-tagged aliases for the stubborn fmt0p1 trajectory. gs100 deliberately resolves to the same
    # dir as the untagged key above, so an OOD study can name cells like the in-domain one
    # (…-fmt0p1-gs{50,100}) while the untagged key keeps earlier studies on their result dirs.
    qwen3-4B-rl-stubborn-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-4B-rl-stubborn-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-4B-instruct-base)               RM_PATH="$TC/Qwen3-4B-Instruct-2507";                            RM_TP=1 ;;
    qwen3-8B-base)                        RM_PATH="$MODELS_DIR/Qwen3-8B";                                  RM_TP=1 ;;
    qwen3-14B-base)                       RM_PATH="$TC/Qwen3-14B";                                         RM_TP=1 ;;
    qwen3.5-35B)                          RM_PATH="$TC/Qwen3.5-35B-A3B";                                   RM_TP=2 ;;
    *)                                    RM_PATH="";                                                      RM_TP=1 ;;
  esac
}

resolve_model "$SENDER";   SPATH="${SENDER_MODEL_PATH:-$RM_PATH}";   STP="${SENDER_TP:-$RM_TP}"
resolve_model "$RECEIVER"; RPATH="${RECEIVER_MODEL_PATH:-$RM_PATH}"; RTP="${RECEIVER_TP:-$RM_TP}"
[ -n "$SPATH" ] || { echo "[serve] ERROR: unknown SENDER '$SENDER'; pass SENDER_MODEL_PATH."; exit 1; }
[ -n "$RPATH" ] || { echo "[serve] ERROR: unknown RECEIVER '$RECEIVER'; pass RECEIVER_MODEL_PATH."; exit 1; }
[ -d "$SPATH" ] || { echo "[serve] ERROR: SENDER path missing: $SPATH (merge the checkpoint first)."; exit 1; }
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
"$INFER_PYTHON" -c "import openai, agents, rl.monitors, rl.persuasion_interaction" >/dev/null 2>&1 || { echo "[env] ERROR: INFER_PYTHON lacks openai/agents/rl.monitors/rl.persuasion_interaction (verl)."; exit 1; }

# --- boot both SGLang servers (sender on GPU0, receiver/judge=35B on the next GPUs) ---
# Stride 2 so jobs with adjacent ids on one node never collide; modulus 17000 keeps the HTTP port
# <= 55535 (SGLang derives grpc = port + 10000).
SPORT=$(( 20000 + 2 * (SLURM_JOB_ID % 17000) ))
RPORT=$(( SPORT + 1 ))
SSGLOG="logs/sglang_sender_${SLURM_JOB_ID}.log"; RSGLOG="logs/sglang_recv_${SLURM_JOB_ID}.log"
echo "[serve] sender   $SENDER <- $SPATH (TP=$STP, GPU $SGPUS) on :$SPORT"
echo "[serve] receiver $RECEIVER <- $RPATH (TP=$RTP, GPU $RGPUS) on :$RPORT  (judge=$JUDGE)"
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

# --- endpoint routing: the game stage gets the sender endpoint as CLI args; the receiver and the
#     deception judge both use rl.receiver_client (RECEIVER_HOST/PORT/MODEL_ID, the training
#     transport and sampling), so the served receiver judges its own games, as in training, and
#     --judge-model is an accepted no-op. The SGLANG_*_<JUDGE key> exports (upper-cased, non-alnum
#     -> '_') only feed ModelAPI callers, so a manual run with the paper metrics still reaches the
#     local server. ---
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$RPORT" RECEIVER_MODEL_ID="$RECEIVER"
env_key() { printf '%s' "$1" | tr 'a-z' 'A-Z' | sed 's/[^A-Z0-9]/_/g'; }
JKEY="$(env_key "$JUDGE")"
export "SGLANG_HOST_${JKEY}=127.0.0.1" "SGLANG_PORT_${JKEY}=$RPORT"
export SGLANG_ENABLE_THINKING=0
SAMP_RECV='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
export "SGLANG_SAMPLING_${JKEY}=$SAMP_RECV"
export PYTHON="$INFER_PYTHON"

FULL_TEMPLATE="datasets/house_showing/processed/full/house_showing_full.json"
SENDER_SLUG="${SENDER//\//-}"; RECV_SLUG="${RECEIVER//\//-}"
RES_DIR="experiments/results/house-showing/${SENDER_SLUG}"

echo "=== [houseshowing-rl] sender=$SENDER receiver=$RECEIVER judge=$JUDGE profiles='$PROFILES' games=$END_IDX rounds=$NUM_STEPS ==="
for PROFILE in $PROFILES; do
  echo "------------------------------------------------------------------"
  echo "[run] profile=$PROFILE (RL-parity rollout)"
  RES="$RES_DIR/${PROFILE}_house_showing_rlrollout${RES_TAG}__recv_${RECV_SLUG}.json"
  "$INFER_PYTHON" evaluation/rl_rollout.py --receiver-api sglang --domain house-showing --profile "$PROFILE" \
    --sender-host 127.0.0.1 --sender-port "$SPORT" --sender-name "$SENDER" \
    --sender-prompt "$PROMPT" \
    --end-idx "$END_IDX" --num-rounds "$NUM_STEPS" --max-workers "$MAX_WORKERS" \
    --out "$RES"
  [ -f "$RES" ] || { echo "[run] ERROR: result not found: $RES"; ls -la "$RES_DIR" || true; exit 1; }
  echo "[eval] profile=$PROFILE judge=$JUDGE -> ${RES%.json}.deception.json"
  "$INFER_PYTHON" evaluation/house_showing/evaluate_houseshowing.py \
    --result-file "$RES" \
    --full-template "$FULL_TEMPLATE" \
    --model "$SENDER" \
    --receiver-config "$PROFILE" \
    --judge-model "$JUDGE" \
    --max-workers "$MAX_WORKERS" \
    --no-wandb
done

echo "[done] house-showing generalization eval complete for sender=$SENDER."
echo "       metrics: $RES_DIR/<profile>_house_showing_rlrollout${RES_TAG}__recv_${RECV_SLUG}.deception.json"
