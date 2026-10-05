#!/bin/bash
#SBATCH --job-name=ob_strat_sweep
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rollout stage of the reward landscape (Old Bailey). A stock prosecutor sender plays the 100
# held-out games against the served Qwen3.5-35B juror once per prompt arm: base (plain role
# prompt), strategies (42-technique guide), and one single_strategy:<slug> arm per taxonomy
# technique ("the only technique you may use is <technique>"). Two senders x 44 arms against the
# stubborn juror = the 88 cells the figure correlates; only the prompt differs between cells.
#
# Game driver: evaluation/rl_rollout.py, the RL-parity rollout (multi-turn sender chat, verl sender
# sampling, receiver via rl.cognitive_models.build_receiver_prompt with the bare-number belief
# _instr + rl.receiver_client RL sampling, val split bound to the training parquet).
# --sender-prompt is the one seam. wandb is off; per-cell numbers land in <result>.json.metrics.json.
#
# Co-serves the sender (TP=1, GPU 0) and the 35B juror (TP=2, GPU 1-2; needs an 80 GB card). This
# launcher takes SGLang senders only. For a hosted sender, serve the juror with
# scripts/serve_receiver_xnode.slurm and drive evaluation/rl_rollout.py --sender-api from a machine
# with API access.
#
# Runs PROMPTS x RECEIVERS (juror profiles) cells per SENDER. The 42 single-technique specs:
#   ONLY=$(.venv/bin/python -c "from rl.strategy_audit.taxonomy import slugs; print(' '.join('single_strategy:'+s for s in slugs()))")
#
#   # smoke (2 games, 1 round, 2 cells):
#   sbatch --export=ALL,SENDER=qwen3-4B-instruct-base,SMOKE=1 evaluation/reward_landscape/rollout_sweep.slurm
#   # the released grid (2 senders x 44 arms x stubborn), one job per sender:
#   for S in qwen3-4B-instruct-base qwen3-8B-base; do
#     sbatch --export=ALL,SENDER=$S,PROMPTS="base strategies $ONLY" evaluation/reward_landscape/rollout_sweep.slurm; done
#
# Required: SENDER (a served sender model key; see resolve_model, or pass SENDER_MODEL_PATH).
# Optional: RECEIVERS("stubborn"; juror profiles, one cell each)
#           PROMPTS("strategies base"; sender prompt specs, base | strategies |
#           single_strategy:<slug>, one cell each; rl/sender_prompts.py)
#           RESULTS_ROOT NUM_STEPS(3) END_IDX(all val) MAX_WORKERS(16) AUDIT_N_GAMES(100)
#           AUDIT_WORKERS(256) SKIP(false) SMOKE(0) SENDER_TP(1) RECEIVER_TP(2)
#           SENDER_MODEL_PATH/RECEIVER_MODEL_PATH/SGLANG_EXTRA VAL_PARQUET SGLANG_VENV
#           INFER_PYTHON REPO MODELS_DIR(<repo>/models) CACHE_DIR(<repo>/.cache)
set -euo pipefail

: "${SENDER:?Must set SENDER (a served sender model key; see resolve_model)}"
# The juror is fixed: the figure correlates technique use with this juror's belief and the audit
# judge is the same model. Any other RECEIVER is rejected by the juror guard below (it needs the
# repo cwd and the venv python).
JUROR="qwen3.5-35B"
_RECEIVER_REQ="${RECEIVER:-$JUROR}"
RECEIVER="$JUROR"
JUDGE="$RECEIVER"
SMOKE="${SMOKE:-0}"
PROMPTS="${PROMPTS:-strategies base}"   # sender prompt specs, one cell each
RECEIVERS="${RECEIVERS:-stubborn}"
NUM_STEPS="${NUM_STEPS:-3}"
END_IDX="${END_IDX:-}"                 # cap after split selection; empty = all 100 val games
MAX_WORKERS="${MAX_WORKERS:-16}"
AUDIT_N_GAMES="${AUDIT_N_GAMES:-100}"
AUDIT_WORKERS="${AUDIT_WORKERS:-256}"
SKIP="${SKIP:-false}"
if [ "$SMOKE" = 1 ]; then
  # Runs every per-cell step in minutes; too thin to estimate anything (analysis stops at the stats json).
  PROMPTS="strategies single_strategy:anchoring"
  RECEIVERS="stubborn"
  END_IDX=2; NUM_STEPS=1; MAX_WORKERS=2; AUDIT_N_GAMES=2
  echo "[smoke] SMOKE=1 -> prompts='$PROMPTS' receivers='$RECEIVERS' end_idx=$END_IDX rounds=$NUM_STEPS audit_n=$AUDIT_N_GAMES"
fi
# REPO roots the venv and config (REPO=/abs/path to relocate); RESULTS_ROOT holds every rollout,
# audit and figure of the package.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO"; mkdir -p logs
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/experiments/results}"
# The 100 val ids come from the trainer's own parquet (split_rl_sft.py writes one per receiver
# distribution under datasets/old_bailey/_generated/rl/<dist>/). Val ids are the same across
# distributions, so VAL_PARQUET can point at any of them.
VAL_PARQUET="${VAL_PARQUET:-$REPO/datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet}"
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"

# --- juror guard (config/receiver/model/, evaluation/receiver_models.py) ---
# RECEIVER may only name the served juror. A hosted juror is pointed to
# scripts/hosted_juror_login_driver.sh and any other name gets this experiment's error. Any other
# exit from the resolver means the interpreter or module is broken.
_RECV_TRANSPORT="$("$SGLANG_PY" -m evaluation.receiver_models --receiver "$_RECEIVER_REQ" --print-transport)" && _RECV_RC=0 || _RECV_RC=$?
case "${_RECV_TRANSPORT}:${_RECV_RC}" in
  sglang:0|unknown:3) ;;
  gateway:0)
    echo "ERROR: RECEIVER=$_RECEIVER_REQ is a hosted juror (API gateway), and this job only serves local jurors. Play it from a machine with API access: scripts/hosted_juror_login_driver.sh (sender served by scripts/serve_sender_xnode.slurm, judge by scripts/serve_receiver_xnode.slurm)."
    exit 1 ;;
  *)
    echo "[env] ERROR: cannot resolve the juror transport of RECEIVER='$_RECEIVER_REQ': '$SGLANG_PY -m evaluation.receiver_models' printed '${_RECV_TRANSPORT}' and exited $_RECV_RC (is SGLANG_VENV correct?)."
    exit 1 ;;
esac
[ "$_RECEIVER_REQ" = "$JUROR" ] || {
  echo "[serve] ERROR: the reward landscape is defined against the served $JUROR juror; RECEIVER='$_RECEIVER_REQ' is not supported."; exit 1; }

# --- model key -> checkpoint path + tensor-parallel size (base senders + the 35B receiver) ---
# Same keys and paths as scripts/{oldbailey,houseshowing,nutrition}_rl_eval.slurm, so a key is the
# same checkpoint in every launcher.
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
TC="${TC:-$MODELS_DIR}"
resolve_model() {   # $1 = model key -> sets RM_PATH, RM_TP
  case "$1" in
    qwen3-4B-instruct-base) RM_PATH="$TC/Qwen3-4B-Instruct-2507";     RM_TP=1 ;;
    qwen3-8B-base)          RM_PATH="$MODELS_DIR/Qwen3-8B";          RM_TP=1 ;;
    qwen3-14B-base)         RM_PATH="$TC/Qwen3-14B";                  RM_TP=1 ;;
    qwen3.5-35B)            RM_PATH="$TC/Qwen3.5-35B-A3B";            RM_TP=2 ;;
    *)                      RM_PATH="";                               RM_TP=1 ;;
  esac
}

resolve_model "$SENDER";   SPATH="${SENDER_MODEL_PATH:-$RM_PATH}";   STP="${SENDER_TP:-$RM_TP}"
resolve_model "$RECEIVER"; RPATH="${RECEIVER_MODEL_PATH:-$RM_PATH}"; RTP="${RECEIVER_TP:-$RM_TP}"
[ -n "$RPATH" ] || { echo "[serve] ERROR: unknown RECEIVER '$RECEIVER'; pass RECEIVER_MODEL_PATH."; exit 1; }
[ -d "$RPATH" ] || { echo "[serve] ERROR: RECEIVER path missing: $RPATH."; exit 1; }

# The sender must be SGLang-served here (see header).
if [ -z "$SPATH" ]; then
  echo "[serve] ERROR: SENDER '$SENDER' is not an SGLang model. A hosted sender must run from a"
  echo "        machine with API access: scripts/serve_receiver_xnode.slurm + evaluation/rl_rollout.py --sender-api."
  exit 1
fi
[ -d "$SPATH" ] || { echo "[serve] ERROR: SENDER path missing: $SPATH."; exit 1; }

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
NEED=$(( STP + RTP ))
if (( NEED > NGPU )); then
  echo "[serve] ERROR: need $NEED GPU(s) (sender TP=$STP + receiver TP=$RTP) but only $NGPU visible. Resubmit with --gres=gpu:${NEED}."
  exit 1
fi
SGPUS="$(seq -s, 0 $((STP-1)))"; RGPUS="$(seq -s, "$STP" $((STP+RTP-1)))"

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
# rl_rollout needs verl importable (rl.persuasion_interaction) + pandas (--val-parquet).
"$INFER_PYTHON" -c "import openai, pandas, agents, rl.strategy_audit.run, rl.receiver_client, rl.persuasion_interaction" >/dev/null 2>&1 \
  || { echo "[env] ERROR: INFER_PYTHON lacks openai/pandas/agents/rl.* (verl importable?)."; exit 1; }
[ -f "$VAL_PARQUET" ] || { echo "[env] ERROR: VAL_PARQUET missing: $VAL_PARQUET (datasets/old_bailey/split_rl_sft.py writes it; or pass VAL_PARQUET=<parquet>)"; exit 1; }

# --- boot the two SGLang servers: the sender and the receiver/judge 35B ---
SPORT=$(( 20000 + 2 * (SLURM_JOB_ID % 17000) ))
RPORT=$(( SPORT + 1 ))
SSGLOG="logs/sglang_sender_${SLURM_JOB_ID}.log"; RSGLOG="logs/sglang_recv_${SLURM_JOB_ID}.log"
echo "[serve] sender   $SENDER <- $SPATH (TP=$STP, GPU $SGPUS) on :$SPORT"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES="$SGPUS" \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$SPATH" --served-model-name "$SENDER" \
    --host 127.0.0.1 --port "$SPORT" --tp-size "$STP" --reasoning-parser qwen3 \
    ${SGLANG_EXTRA:-} > "$SSGLOG" 2>&1 &
SPID=$!
echo "[serve] receiver $RECEIVER <- $RPATH (TP=$RTP, GPU $RGPUS) on :$RPORT  (judge=$JUDGE)"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES="$RGPUS" \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$RPATH" --served-model-name "$RECEIVER" \
    --host 127.0.0.1 --port "$RPORT" --tp-size "$RTP" --reasoning-parser qwen3 \
    ${SGLANG_EXTRA:-} > "$RSGLOG" 2>&1 &
RPID=$!
cleanup() { kill "$RPID" "$SPID" 2>/dev/null || true; }
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
wait_ready "$SPORT" "$SPID" "$SSGLOG" "sender" "$SENDER" || exit 1
wait_ready "$RPORT" "$RPID" "$RSGLOG" "receiver" "$RECEIVER" || exit 1

# --- per-model endpoint routing (agents/model/openai_llm._sglang_endpoint): KEY = model_id upper-cased,
#     non-alnum -> '_'. Game sender and receiver go through ModelAPI/SGLangModel, the audit judge
#     through rl.receiver_client (RECEIVER_HOST/PORT/MODEL_ID). RL-matched sampling, <think> off. ---
env_key() { printf '%s' "$1" | tr 'a-z' 'A-Z' | sed 's/[^A-Z0-9]/_/g'; }
RKEY="$(env_key "$RECEIVER")"
SAMP_POLICY='{"temperature":1.0,"top_p":1.0}'
SAMP_RECV='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
SKEY="$(env_key "$SENDER")"
export "SGLANG_HOST_${RKEY}=127.0.0.1" "SGLANG_PORT_${RKEY}=$RPORT" "SGLANG_SAMPLING_${RKEY}=$SAMP_RECV"
export "SGLANG_HOST_${SKEY}=127.0.0.1" "SGLANG_PORT_${SKEY}=$SPORT" "SGLANG_SAMPLING_${SKEY}=$SAMP_POLICY"
export SGLANG_ENABLE_THINKING=0
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$RPORT" RECEIVER_MODEL_ID="$RECEIVER"
export PYTHON="$INFER_PYTHON"

SENDER_SLUG="${SENDER//\//-}"; RECV_SLUG="${RECEIVER//\//-}"
RES_DIR="$RESULTS_ROOT/old-bailey/${SENDER_SLUG}"

echo "=== [reward-landscape/rl-parity] sender=$SENDER receiver=$RECEIVER judge=$JUDGE prompts='$PROMPTS' receivers='$RECEIVERS' rounds=$NUM_STEPS end_idx=${END_IDX:-all-val} audit_n=$AUDIT_N_GAMES ==="
for PROMPT in $PROMPTS; do
  # Validate the spec, then take the result-file stem and arm key from the resolver: the ':' in
  # single_strategy:framing must never reach a path, and result manifests are keyed by arm key.
  "$INFER_PYTHON" -m rl.sender_prompts --check "$PROMPT" >/dev/null || {
    echo "[cell] ERROR: unknown PROMPT '$PROMPT' (base | strategies | single_strategy:<slug>; slugs: rl/strategy_audit/taxonomy.py)"; exit 1; }
  SENDER_STEM="$("$INFER_PYTHON" -m rl.sender_prompts --print result-stem --prompt "$PROMPT")" || {
    echo "[cell] ERROR: could not derive the result stem for PROMPT '$PROMPT'"; exit 1; }
  ARM_KEY="$("$INFER_PYTHON" -m rl.sender_prompts --print arm-key --prompt "$PROMPT")" || {
    echo "[cell] ERROR: could not derive the arm key for PROMPT '$PROMPT'"; exit 1; }
  for PROFILE in $RECEIVERS; do
    case "$PROFILE" in
      neutral)  PRIOR=0.5 ;;
      stubborn) PRIOR=0.1 ;;
      *) echo "[cell] ERROR: unknown RECEIVER profile '$PROFILE' (expected: neutral stubborn)"; exit 1 ;;
    esac
    echo "------------------------------------------------------------------"
    echo "[cell] prompt=$PROMPT ($SENDER_STEM)  receiver=$PROFILE (prior=$PRIOR)"
    RES="${RES_DIR}/${PROFILE}_oldbailey_rlrollout_val_sender-${SENDER_STEM}__recv_${RECV_SLUG}.json"
    SKIPFLAG=$([ "$SKIP" = true ] && echo "--skip" || echo "--no-skip")
    # Game (RL-parity rollout).
    "$INFER_PYTHON" evaluation/rl_rollout.py --receiver-api sglang --domain old-bailey --profile "$PROFILE" \
      --sender-host 127.0.0.1 --sender-port "$SPORT" --sender-name "$SENDER" \
      --sender-prompt "$PROMPT" \
      --val-parquet "$VAL_PARQUET" --num-rounds "$NUM_STEPS" --max-workers "$MAX_WORKERS" \
      ${END_IDX:+--end-idx "$END_IDX"} "$SKIPFLAG" --out "$RES"
    [ -f "$RES" ] || { echo "[cell] ERROR: result not found: $RES"; ls -la "$RES_DIR" 2>/dev/null || true; exit 1; }
    # Persuasion metrics: final P(guilty) against the profile's prior.
    echo "[eval] -> ${RES}.metrics.json  (prior=$PRIOR)"
    "$INFER_PYTHON" evaluation/old_bailey/evaluate_oldbailey.py \
      --result-file "$RES" --prior "$PRIOR" \
      --model "$SENDER" --receiver-config "$PROFILE" --sender-strategy "$ARM_KEY" --no-wandb
    # Per-cell 42-way audit, for reading one cell on its own. Not the figure's instrument: the
    # correlation uses reaudit_all.slurm's audits, which judge the whole grid with one judge instance.
    echo "[audit] -> ${RES%.json}.strategy_audit.json  (judge=$JUDGE, n=$AUDIT_N_GAMES)"
    HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= \
      "$INFER_PYTHON" -m rl.strategy_audit.run --oldbailey-result "$RES" \
        --n-games "$AUDIT_N_GAMES" --max-workers "$AUDIT_WORKERS"
  done
done

echo "[done] rollout sweep complete for sender=$SENDER."
echo "       transcripts: ${RES_DIR}/<profile>_oldbailey_rlrollout_val_sender-<stem>__recv_${RECV_SLUG}.json"
echo "       persuasion:  ${RES_DIR}/<profile>_oldbailey_rlrollout_val_sender-<stem>__recv_${RECV_SLUG}.json.metrics.json"
echo "       next:        build_manifest.py -> reaudit_all.slurm -> run.sh -> plot_belief_correlation.py"
