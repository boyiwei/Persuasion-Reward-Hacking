#!/bin/bash
#SBATCH --job-name=sof_probe
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stage C of the source-of-fabrication experiment, one job per (SIZE, probe checkpoint).
# Serves one policy on 1 GPU (no juror) and asks it, for every item of the item set, whether the
# statement is true evidence: --context-mode out (Direct Ask) and the in-context arms. A cross-node
# mkdir lock makes concurrent or resubmitted jobs safe, the item table's build id guards resume, and
# the prober's checkpoint fingerprint is asserted against the one stage A recorded.
#
#   sbatch --export=ALL,SIZE=8B,PCKPT=gs100 evaluation/source_of_fabrication/probe_cell.slurm
#   sbatch --export=ALL,SIZE=8B,PCKPT=gs100,SMOKE=1 <this file>
set -euo pipefail

: "${SIZE:?Must set SIZE (4B|8B)}"
: "${PCKPT:?Must set PCKPT — the PROBE checkpoint (base|gs50|gs100)}"
case "$SIZE" in 4B|8B) ;; *) echo "[env] ERROR: SIZE must be 4B|8B"; exit 1 ;; esac
case "$PCKPT" in base|gs50|gs100) ;; *) echo "[env] ERROR: PCKPT must be base|gs50|gs100"; exit 1 ;; esac
SMOKE="${SMOKE:-0}"
TAG="${TAG:-}"; if [ "$SMOKE" = 1 ] && [ -z "$TAG" ]; then TAG="_smoke"; fi
PROBES="${PROBES:-out in_pre_role}"            # add in_pre_honest for the honest-pause arm
if [ "$SMOKE" = 1 ]; then
  N_DRAWS="${N_DRAWS:-2}"; MAX_WORKERS="${MAX_WORKERS:-4}"
fi
N_DRAWS="${N_DRAWS:-3}"                       # draw #1 = readout, #2.. = stability
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_WORKERS="${MAX_WORKERS:-24}"
ITEMS_FILE="${ITEMS_FILE:-}"                  # explicit item-file override
CONTEXTS_FILE="${CONTEXTS_FILE:-}"            # its context table

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"; mkdir -p logs
SOF_DIR="$REPO/evaluation/source_of_fabrication"
[ -f "$SOF_DIR/run_probe.py" ] || { echo "[env] ERROR: package dir not found: $SOF_DIR"; exit 1; }
SOF_ROOT="${SOF_ROOT:-$REPO/experiments/results/source_of_fabrication}"
ITEMS_DIR="${ITEMS_DIR:-$SOF_ROOT/probe_items_union${TAG}}"
ANS_DIR="$SOF_ROOT/answers_union${TAG}"
LOG_DIR="$SOF_ROOT/logs"
case "$ANS_DIR" in "$SOF_DIR"*) echo "[guard] refusing to write results into the package source dir"; exit 1 ;; esac
mkdir -p "$ANS_DIR" "$LOG_DIR"
# shellcheck source=ckpt_paths.sh
source "$SOF_DIR/ckpt_paths.sh"
[ "$RECEIVER_NAME" = "qwen3.5-35B" ] || { echo "[env] ERROR: this experiment's juror is the locally served qwen3.5-35B"; exit 1; }
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"
INFER_PYTHON="${INFER_PYTHON:-$SGLANG_PY}"

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

[[ -x "$SGLANG_PY" ]] || { echo "[env] ERROR: venv python not found at '$SGLANG_PY'"; exit 1; }
"$SGLANG_PY" -c "import sglang" >/dev/null 2>&1 || { echo "[env] ERROR: '$SGLANG_PY' cannot import sglang"; exit 1; }
unset PYTHONPATH || true
"$INFER_PYTHON" -c "import openai, agents, rl.monitors, evaluation.source_of_fabrication.probe_prompts" >/dev/null 2>&1 \
  || { echo "[env] ERROR: INFER_PYTHON lacks openai/agents/rl or cannot import the package"; exit 1; }

# --- the probe prompt is the measuring instrument: fail here if its text moved ------------------------
"$INFER_PYTHON" "$SOF_DIR/probe_prompts.py" --selftest \
  || { echo "[parity] ERROR: the composed probe prompt does not match its pins"; exit 1; }

# --- items ------------------------------------------------------------------------------------------
if [ -n "$ITEMS_FILE" ]; then
  UNION="$ITEMS_FILE"; CTXS="${CONTEXTS_FILE:?CONTEXTS_FILE is required with ITEMS_FILE}"
  BID="$(sha256sum "$UNION" | cut -c1-64)"
else
  UNION="$ITEMS_DIR/union__${SIZE}.jsonl"; CTXS="$ITEMS_DIR/contexts_union__${SIZE}.jsonl.gz"
  [ "$SMOKE" = 1 ] && UNION="$ITEMS_DIR/union_smoke__${SIZE}.jsonl"
  [ -f "$ITEMS_DIR/_union_summary.json" ] || { echo "[items] ERROR: build the item set on a login node first (build_item_set.py)"; exit 1; }
  BID="$("$INFER_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['sizes'][sys.argv[2]]['union_build_id'])" "$ITEMS_DIR/_union_summary.json" "$SIZE")"
fi
for _f in "$UNION" "$CTXS"; do [ -f "$_f" ] || { echo "[items] ERROR: missing $_f"; exit 1; }; done
# Resume guard: answers keyed on item_id are only valid against the item table they were drawn from.
if [ -f "$ANS_DIR/_build_id__${SIZE}" ]; then
  [ "$(cat "$ANS_DIR/_build_id__${SIZE}")" = "$BID" ] \
    || { echo "[resume] ERROR: the item table changed since answers in $ANS_DIR were written"; exit 1; }
else
  echo "$BID" > "$ANS_DIR/_build_id__${SIZE}"
fi

# --- cross-node mutex: mkdir (atomic on a shared filesystem; flock is per-node) + one EXIT trap --
SPID=""
LOCKDIR="$ANS_DIR/.lock__${SIZE}__${PCKPT}.d"
cleanup() { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null || true; rm -rf "$LOCKDIR"; }
if mkdir "$LOCKDIR" 2>/dev/null; then
  printf '%s %s %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)" > "$LOCKDIR/owner"
  trap cleanup EXIT; trap 'exit 143' TERM INT   # scancel -> EXIT trap -> lock removed
else
  jid="$(cut -d' ' -f1 "$LOCKDIR/owner" 2>/dev/null || true)"
  if [ -n "$jid" ] && squeue -h -j "$jid" 2>/dev/null | grep -q .; then
    echo "[lock] ${SIZE}/${PCKPT} is being probed by live job $jid — exiting"; exit 0
  fi
  echo "[lock] ERROR: STALE lock $LOCKDIR from dead job '${jid:-?}' — rm -rf it and resubmit"; exit 1
fi

resolve_ckpt "$SIZE" "$PCKPT"; SPATH="${SENDER_MODEL_PATH:-$CK_PATH}"; SNAME="$CK_NAME"
[ -d "$SPATH" ] || { echo "[serve] ERROR: prober path missing: $SPATH"; exit 1; }
echo "[serve] prober path: $SPATH"
# The effective chat template of the directory this job actually serves, recorded next to its
# answers; check_templates.py --verify-dir applies the cross-prober identity gate to the records.
"$INFER_PYTHON" "$SOF_DIR/check_templates.py" --sizes "$SIZE" --ckpts "$PCKPT" --path "$SPATH" \
  --out "$ANS_DIR/template_check_${SIZE}_${PCKPT}.json" \
  || { echo "[tpl] ERROR: cannot read the chat template of $SPATH"; exit 1; }
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
(( NGPU >= 1 )) || { echo "[serve] ERROR: need 1 GPU (prober TP=1), have $NGPU"; exit 1; }
# A100 has no FA3: use flashinfer there; H100/H200 keep the default backend.
if nvidia-smi -L | grep -q "A100" && [[ "${SGLANG_EXTRA:-}" != *attention-backend* ]]; then
  SGLANG_EXTRA="${SGLANG_EXTRA:-} --attention-backend flashinfer"
fi
PORT_BASE=$(( 21000 + 2 * (SLURM_JOB_ID % 17000) )); SPORT="$PORT_BASE"
echo "=== [sof-probe] SIZE=$SIZE PROBE=$PCKPT probes='$PROBES' draws=$N_DRAWS tag='$TAG' items=$(basename "$UNION") (GPUs=$NGPU, extra='${SGLANG_EXTRA:-}') ==="

# --- fingerprint (background, while the server loads): assert the stage-A weights -------------------
FP="$ANS_DIR/_ckpt__${SIZE}__${PCKPT}.json"
FP_ARGS=(--path "$SPATH" --out "$FP" --label "${SIZE}/${PCKPT}")
EXPECT_FP="${EXPECT_FP:-}"
if [ -z "$EXPECT_FP" ]; then
  for _cand in "$SOF_ROOT/rollouts/_ckpt__${SIZE}__${PCKPT}.json" "$SOF_ROOT/rollouts${TAG}/_ckpt__${SIZE}__${PCKPT}.json"; do
    [ -f "$_cand" ] && { EXPECT_FP="$_cand"; break; }
  done
fi
[ -n "$EXPECT_FP" ] && FP_ARGS+=(--expect "$EXPECT_FP")
"$INFER_PYTHON" "$SOF_DIR/ckpt_fingerprint.py" "${FP_ARGS[@]}" > "$LOG_DIR/fp_probe_${SLURM_JOB_ID}.log" 2>&1 & FP_PID=$!

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
launch_server "$SPATH" "$SNAME" 1 "0" "$SPORT" "$LOG_DIR/sg_probe_${SLURM_JOB_ID}.log"; SPID=$_PID
wait_ready "$SPORT" "$SPID" "$LOG_DIR/sg_probe_${SLURM_JOB_ID}.log" "prober" "$SNAME" || exit 1
wait "$FP_PID" || { echo "[fp] ERROR: checkpoint fingerprint/expect check FAILED:"; cat "$LOG_DIR/fp_probe_${SLURM_JOB_ID}.log"; exit 1; }
cat "$LOG_DIR/fp_probe_${SLURM_JOB_ID}.log"

# --- the probes --------------------------------------------------------------------------------------
unresolved_errors() {  # $1=file -> number of item_ids whose last row is an __ERROR__ readout
  "$INFER_PYTHON" - "$1" <<'PYEOF'
import json, sys
last = {}
for ln in open(sys.argv[1]):
    if ln.strip():
        r = json.loads(ln); last[r["item_id"]] = str(r.get("readout_response", "")).startswith("__ERROR__")
print(sum(last.values()))
PYEOF
}
N_EXP="$(wc -l < "$UNION")"
PROBE_ARGS=(--host 127.0.0.1 --port "$SPORT" --served-name "$SNAME" --items "$UNION"
            --n-draws "$N_DRAWS" --max-tokens "$MAX_TOKENS" --max-workers "$MAX_WORKERS")
for PROBE in $PROBES; do
  OUT="$ANS_DIR/${SIZE}__P${PCKPT}__${PROBE}.jsonl"
  echo "[C] probe=$PROBE prober=$SNAME -> $(basename "$OUT")"
  case "$PROBE" in
    out)
      "$INFER_PYTHON" "$SOF_DIR/run_probe.py" "${PROBE_ARGS[@]}" --context-mode out --out "$OUT" ;;
    in_pre_honest|in_pre_role)
      ARM="${PROBE##*_}"
      "$INFER_PYTHON" "$SOF_DIR/run_probe.py" "${PROBE_ARGS[@]}" \
        --context-mode in --arm "$ARM" --assertion pre --contexts "$CTXS" --out "$OUT" ;;
    *) echo "[C] ERROR: unknown probe '$PROBE'"; exit 1 ;;
  esac
  N_ROWS="$(wc -l < "$OUT")"; N_ERR="$(unresolved_errors "$OUT")"
  echo "[post] $(basename "$OUT"): $N_ROWS rows (expected >= $N_EXP), $N_ERR unresolved __ERROR__ items"
  [ "$N_ROWS" -ge "$N_EXP" ] || { echo "[post] ERROR: fewer rows than items"; exit 1; }
  [ "$N_ERR" = 0 ] || { echo "[post] ERROR: $N_ERR items still end in __ERROR__ — resubmit (resume retries only those)"; exit 1; }
done
echo "[sof-probe] complete: SIZE=$SIZE PROBE=$PCKPT -> $ANS_DIR/${SIZE}__P${PCKPT}__*.jsonl (fingerprint $FP)"
