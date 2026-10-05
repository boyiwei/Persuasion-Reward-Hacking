#!/bin/bash
#SBATCH --job-name=sof_items
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stages A and B of the source-of-fabrication experiment, one job per (SIZE, generation checkpoint).
# Serves the sender checkpoint (TP=1, GPU0) and the 35B juror/judge (TP=2, GPU1-2; needs an 80 GB
# card) and runs the per-cell pipeline in one allocation; every stage is resume-safe:
#   A1  evaluation/rl_rollout.py    play the 100 held-out val games (RL-parity replay, base prompt,
#                                   stubborn juror)
#   A2  audit_rollouts.py           the fabrication judge, AUDIT_PASSES passes so judge_fail games
#                                   are retried in-job
#   B   build_probe_items.py        round attribution + context reconstruction (CPU)
#   B'  build_item_set.py --emit-candidates   namespaced candidates, for a cell drawn more than once
# The sender's checkpoint fingerprint (sha256 of every file) is written first so the probe stage can
# assert it serves byte-identical weights.
#
# DRAW=<name> is an extra sampling draw of the same cell: its items go to probe_items/<draw>/ and
# build_item_set.py pools them with the primary draw. Leave it empty for the primary draw.
#
#   sbatch --export=ALL,SIZE=8B,CKPT=base evaluation/source_of_fabrication/generate_items.slurm
#   sbatch --export=ALL,SIZE=8B,CKPT=base,DRAW=d2 <this file>        # an extra draw of that cell
#   sbatch --export=ALL,SIZE=8B,CKPT=base,SMOKE=1 <this file>        # 3 games, *_smoke dirs
set -euo pipefail

: "${SIZE:?Must set SIZE (4B|8B)}"
: "${CKPT:?Must set CKPT — the GENERATION checkpoint (base|gs50|gs100)}"
case "$SIZE" in 4B|8B) ;; *) echo "[env] ERROR: SIZE must be 4B|8B"; exit 1 ;; esac
case "$CKPT" in base|gs50|gs100) ;; *) echo "[env] ERROR: CKPT must be base|gs50|gs100"; exit 1 ;; esac
DRAW="${DRAW:-}"                              # extra sampling draw of this cell, e.g. d2
SMOKE="${SMOKE:-0}"
TAG="${TAG:-}"; if [ "$SMOKE" = 1 ] && [ -z "$TAG" ]; then TAG="_smoke"; fi
if [ "$SMOKE" = 1 ]; then
  END_IDX="${END_IDX:-3}"; AUDIT_PASSES="${AUDIT_PASSES:-1}"; MAX_WORKERS="${MAX_WORKERS:-4}"
fi
END_IDX="${END_IDX:-0}"                       # cap games after split selection (0 = all)
MAX_WORKERS="${MAX_WORKERS:-16}"
PER_GAME_CAP="${PER_GAME_CAP:-3}"             # fabrications kept per game
AUDIT_PASSES="${AUDIT_PASSES:-3}"             # audit_rollouts.py retries judge_fail games on resume
SENDER_PROMPT="${SENDER_PROMPT:-base}"        # the heatmap generates under the plain role prompt

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"; mkdir -p logs
SOF_DIR="$REPO/evaluation/source_of_fabrication"
[ -f "$SOF_DIR/build_item_set.py" ] || { echo "[env] ERROR: package dir not found: $SOF_DIR"; exit 1; }
SOF_ROOT="${SOF_ROOT:-$REPO/experiments/results/source_of_fabrication}"
case "$SOF_ROOT" in "$SOF_DIR"*) echo "[guard] refusing to write results into the package source dir"; exit 1 ;; esac
# shellcheck source=ckpt_paths.sh
source "$SOF_DIR/ckpt_paths.sh"
VAL_PARQUET="${VAL_PARQUET:-$REPO/datasets/old_bailey/_generated/rl/${PROFILE}/rl_validation.parquet}"

SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"
INFER_PYTHON="${INFER_PYTHON:-$SGLANG_PY}"
[[ -x "$SGLANG_PY" ]] || { echo "[env] ERROR: venv python not found at '$SGLANG_PY'"; exit 1; }

# --- juror guard (config/receiver/model/, evaluation/receiver_models.py) ---
# The served 35B is both game receiver and stage-A2 fabrication judge, so RECEIVER_NAME may only
# name it. A hosted juror gets a redirect; any other exit means the interpreter or module is
# broken. Runs before the output dirs, resume check and serving, so a bad juror is always refused.
_RECV_TRANSPORT="$("$SGLANG_PY" -m evaluation.receiver_models --receiver "$RECEIVER_NAME" --print-transport)" && _RECV_RC=0 || _RECV_RC=$?
case "${_RECV_TRANSPORT}:${_RECV_RC}" in
  sglang:0|unknown:3) ;;
  gateway:0)
    echo "ERROR: RECEIVER=$RECEIVER_NAME is a hosted juror (API gateway), and this job only serves local jurors. Play it from a machine with API access: scripts/hosted_juror_login_driver.sh (sender served by scripts/serve_sender_xnode.slurm, judge by scripts/serve_receiver_xnode.slurm)."
    exit 1 ;;
  *)
    echo "[env] ERROR: cannot resolve the juror transport of RECEIVER_NAME='$RECEIVER_NAME': '$SGLANG_PY -m evaluation.receiver_models' printed '${_RECV_TRANSPORT}' and exited $_RECV_RC (is SGLANG_VENV correct?)."
    exit 1 ;;
esac
[ "$RECEIVER_NAME" = "qwen3.5-35B" ] || { echo "[env] ERROR: this experiment's juror is the locally served qwen3.5-35B"; exit 1; }

ROLL_DIR="$SOF_ROOT/rollouts${TAG}"; AUD_DIR="$SOF_ROOT/audits${TAG}"
RAW_DIR="$SOF_ROOT/probe_items${TAG}/${DRAW:-primary}"
LOG_DIR="$SOF_ROOT/logs"
mkdir -p "$ROLL_DIR" "$AUD_DIR" "$RAW_DIR" "$LOG_DIR"
STEM="${SIZE}__${CKPT}${DRAW:+__${DRAW}}"
RES="$ROLL_DIR/rollout__${STEM}.json"
AUD="$AUD_DIR/audit__${STEM}.jsonl"
FP="$ROLL_DIR/_ckpt__${SIZE}__${CKPT}.json"
ITEMS="$RAW_DIR/incontext__${SIZE}__${CKPT}.jsonl"; CTXS="$RAW_DIR/contexts__${SIZE}__${CKPT}.jsonl.gz"
CAND="$RAW_DIR/candidates__${SIZE}__${CKPT}.jsonl"
DONE_MARK="$ITEMS"
if [ -n "$DRAW" ]; then DONE_MARK="$CAND"; fi
if [ -s "$ITEMS" ] && [ -s "$CTXS" ] && [ -s "$DONE_MARK" ] && [ -s "$FP" ]; then
  echo "[items] complete output exists: $DONE_MARK — nothing to do"; exit 0
fi

# --- network / caches / JIT toolchain ---------------------------------------------------------------
module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export WANDB_MODE=offline
_CUDA_SYS="${CUDA_SYS:-/usr/local/cuda-13.0}"
_CUDA_INC="$SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13/include"
export CUDA_HOME="${CUDA_HOME:-$CACHE_DIR/cuda13-merged}"; mkdir -p "$CUDA_HOME"
ln -sfn "$_CUDA_SYS/bin" "$CUDA_HOME/bin"; ln -sfn "$_CUDA_INC" "$CUDA_HOME/include"; ln -sfn "$_CUDA_SYS/lib64" "$CUDA_HOME/lib64"
export PATH="$SGLANG_VENV/bin:$CUDA_HOME/bin:$PATH"; export CPATH="$_CUDA_INC:${CPATH:-}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$CACHE_DIR/tvm-ffi}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_DIR/flashinfer-base}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

"$SGLANG_PY" -c "import sglang" >/dev/null 2>&1 || { echo "[env] ERROR: '$SGLANG_PY' cannot import sglang"; exit 1; }
unset PYTHONPATH || true
"$INFER_PYTHON" -c "import openai, agents, rl.monitors, rl.receiver_client, evaluation.source_of_fabrication.probe_prompts" >/dev/null 2>&1 \
  || { echo "[env] ERROR: INFER_PYTHON lacks openai/agents/rl or cannot import the package"; exit 1; }
[ -f "$VAL_PARQUET" ] || { echo "[env] ERROR: val parquet missing: $VAL_PARQUET (datasets/old_bailey/split_rl_sft.py writes it)"; exit 1; }
"$INFER_PYTHON" "$SOF_DIR/probe_prompts.py" --selftest \
  || { echo "[parity] ERROR: the composed probe prompt does not match its pins"; exit 1; }

# --- cross-node mutex (mkdir is atomic on a shared filesystem; flock is per-node) + one EXIT trap -
SPID=""; RPID=""
LOCKDIR="$ROLL_DIR/.lock__${STEM}.d"
cleanup() { for p in "${SPID:-}" "${RPID:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; rm -rf "$LOCKDIR"; }
if mkdir "$LOCKDIR" 2>/dev/null; then
  printf '%s %s %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)" > "$LOCKDIR/owner"
  trap cleanup EXIT; trap 'exit 143' TERM INT   # scancel -> EXIT trap -> lock removed
else
  jid="$(cut -d' ' -f1 "$LOCKDIR/owner" 2>/dev/null || true)"
  if [ -n "$jid" ] && squeue -h -j "$jid" 2>/dev/null | grep -q .; then
    echo "[lock] $STEM is being generated by live job $jid — exiting"; exit 0
  fi
  echo "[lock] ERROR: STALE lock $LOCKDIR from dead job '${jid:-?}' — rm -rf it and resubmit"; exit 1
fi

resolve_ckpt "$SIZE" "$CKPT"; SPATH="${SENDER_MODEL_PATH:-$CK_PATH}"; SNAME="$CK_NAME"
[ -d "$SPATH" ] || { echo "[serve] ERROR: sender path missing: $SPATH"; exit 1; }
[ -d "$RECEIVER_PATH" ] || { echo "[serve] ERROR: 35B path missing: $RECEIVER_PATH"; exit 1; }
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
(( NGPU >= 3 )) || { echo "[serve] ERROR: need 3 GPUs (sender TP=1 + 35B TP=2), have $NGPU"; exit 1; }
# A100 has no FA3: use flashinfer there; H100/H200 keep the default backend.
if nvidia-smi -L | grep -q "A100" && [[ "${SGLANG_EXTRA:-}" != *attention-backend* ]]; then
  SGLANG_EXTRA="${SGLANG_EXTRA:-} --attention-backend flashinfer"
fi
PORT_BASE=$(( 21000 + 2 * (SLURM_JOB_ID % 17000) )); SPORT="$PORT_BASE"; RPORT=$(( PORT_BASE + 1 ))
echo "=== [sof-items] SIZE=$SIZE CKPT=$CKPT draw='${DRAW:-primary}' tag='$TAG' prompt=$SENDER_PROMPT end_idx=$END_IDX audit_passes=$AUDIT_PASSES (GPUs=$NGPU, extra='${SGLANG_EXTRA:-}') ==="
echo "[serve] sender path: $SPATH"

# --- fingerprint the sender in the background while the servers load --------------------------------
"$INFER_PYTHON" "$SOF_DIR/ckpt_fingerprint.py" --path "$SPATH" --out "$FP" --label "${SIZE}/${CKPT}" \
  > "$LOG_DIR/fp_items_${SLURM_JOB_ID}.log" 2>&1 & FP_PID=$!

launch_server() {  # $1=path $2=served-name $3=tp $4=gpus $5=port $6=log  -> sets _PID
  echo "[serve] $2 <- $1 (TP=$3, GPU $4) on :$5"
  HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES="$4" \
    "$SGLANG_PY" -m sglang.launch_server --model-path "$1" --served-model-name "$2" \
      --host 127.0.0.1 --port "$5" --tp-size "$3" --reasoning-parser qwen3 \
      ${SGLANG_EXTRA:-} > "$6" 2>&1 &
  _PID=$!
}
wait_ready() {  # $1=port $2=pid $3=log $4=name $5=expected-served-name
  for _i in $(seq 1 240); do
    if ! kill -0 "$2" 2>/dev/null; then echo "[serve] ERROR: $4 died. Tail of $3:"; tail -n 80 "$3"; return 1; fi
    got="$(curl --noproxy '*' -s "http://127.0.0.1:$1/v1/models" 2>/dev/null \
            | "$INFER_PYTHON" -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
    if [ "$got" = "$5" ]; then echo "[serve] $4 UP ($got on :$1)"; return 0; fi
    if [ -n "$got" ]; then echo "[serve] ERROR: :$1 serves '$got' != '$5' (PORT COLLISION)"; tail -n 40 "$3"; return 1; fi
    sleep 5
  done
  echo "[serve] ERROR: $4 not ready. Tail of $3:"; tail -n 120 "$3"; return 1
}
export SGLANG_ENABLE_THINKING=0
launch_server "$SPATH" "$SNAME" 1 "0" "$SPORT" "$LOG_DIR/sg_items_sender_${SLURM_JOB_ID}.log"; SPID=$_PID
launch_server "$RECEIVER_PATH" "$RECEIVER_NAME" 2 "1,2" "$RPORT" "$LOG_DIR/sg_items_35b_${SLURM_JOB_ID}.log"; RPID=$_PID
wait_ready "$SPORT" "$SPID" "$LOG_DIR/sg_items_sender_${SLURM_JOB_ID}.log" "sender" "$SNAME" || exit 1
wait_ready "$RPORT" "$RPID" "$LOG_DIR/sg_items_35b_${SLURM_JOB_ID}.log" "35B" "$RECEIVER_NAME" || exit 1
wait "$FP_PID" || { echo "[fp] ERROR: fingerprint failed:"; cat "$LOG_DIR/fp_items_${SLURM_JOB_ID}.log"; exit 1; }
cat "$LOG_DIR/fp_items_${SLURM_JOB_ID}.log"
# The 35B is the game juror AND the audit judge: point rl.receiver_client at it.
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$RPORT" RECEIVER_MODEL_ID="$RECEIVER_NAME" SGLANG_HOST=127.0.0.1 SGLANG_PORT="$RPORT"

echo "[A1] val games: $SNAME vs $RECEIVER_NAME ($PROFILE) -> $RES"
A1_ARGS=(--receiver-api sglang --domain old-bailey --profile "$PROFILE" --split val --val-parquet "$VAL_PARQUET"
         --sender-prompt "$SENDER_PROMPT"
         --sender-host 127.0.0.1 --sender-port "$SPORT" --sender-name "$SNAME"
         --num-rounds 3 --max-workers "$MAX_WORKERS" --out "$RES")
[ "$END_IDX" -gt 0 ] && A1_ARGS+=(--end-idx "$END_IDX")
"$INFER_PYTHON" evaluation/rl_rollout.py "${A1_ARGS[@]}"
N_COMPLETE="$("$INFER_PYTHON" -c 'import json,sys; g=json.load(open(sys.argv[1])); print(sum(bool(x.get("complete")) for x in g), len(g))' "$RES")"
echo "[A1] complete/total games: $N_COMPLETE"

echo "[A2] fabrication audit ($AUDIT_PASSES passes; resume retries judge_fail games)"
still_failed() {  # $1=audit jsonl -> number of games whose latest record is judge_fail
  "$INFER_PYTHON" - "$1" <<'PYEOF'
import json, sys
last = {}
for ln in open(sys.argv[1]):
    if ln.strip():
        r = json.loads(ln); last[r["game_id"]] = bool(r.get("judge_fail"))
print(sum(last.values()))
PYEOF
}
for _pass in $(seq 1 "$AUDIT_PASSES"); do
  "$INFER_PYTHON" "$SOF_DIR/audit_rollouts.py" --result-file "$RES" --out "$AUD" --max-workers "$MAX_WORKERS"
  NF="$(still_failed "$AUD")"; echo "[A2] pass $_pass: $NF judge_fail games remain"
  [ "$NF" = 0 ] && break
done

echo "[B] round attribution + context reconstruction -> $RAW_DIR"
"$INFER_PYTHON" "$SOF_DIR/build_probe_items.py" --size "$SIZE" --ckpt "$CKPT" \
  --sender-prompt "$SENDER_PROMPT" --result-file "$RES" --audit "$AUD" \
  --per-game-cap "$PER_GAME_CAP" --out-dir "$RAW_DIR"
[ -s "$ITEMS" ] || { echo "[B] ERROR: item table not written: $ITEMS"; exit 1; }

if [ -n "$DRAW" ]; then
  echo "[B'] namespaced candidate items (draw=$DRAW)"
  "$INFER_PYTHON" "$SOF_DIR/build_item_set.py" --emit-candidates --raw-dir "$RAW_DIR" \
    --size "$SIZE" --ckpt "$CKPT" --draw "$DRAW"
  [ -s "$CAND" ] || { echo "[B'] ERROR: candidates file not written: $CAND"; exit 1; }
fi
echo "[sof-items] complete: $(wc -l < "$ITEMS") item rows -> $ITEMS ; fingerprint $FP"
