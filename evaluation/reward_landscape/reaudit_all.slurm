#!/bin/bash
#SBATCH --job-name=reward_landscape_audit
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --signal=B:USR1@600
#
# The figure's audit instrument: every cell of the manifest judged by one served Qwen3.5-35B in one
# process and one job. The figure correlates technique presence with the juror's final belief
# across 88 cells, so a judge seam inside the grid would show up as a technique difference.
#
# Two passes per key, A before B:
#   A  evaluation/audit_fabrications.py --domain old-bailey  -> $BY/<key>.fabrication.json
#   B  python -m rl.strategy_audit.run                       -> $BY/<key>.strategy_audit.json
# B's false_information column comes from A's fabrication count, and run.py resolves the
# --fabrication-json default off the result path, not --out, so an existing sidecar next to a
# result would mix judgings across keys. B therefore always gets --fabrication-json inside
# by_result/. Pass A runs over all keys first so fabrication failures surface before any 42-way
# GPU time is spent.
#
# Both juror profiles (receiver prompt profiles: neutral prior 0.5, stubborn prior 0.1) run in one
# job when the manifest has both; the analysis pools over them, so splitting would add a judge seam.
#
# Serving: one sglang server, tp=2 x dp=2 on 4 GPUs. The server is the bottleneck, so the second
# dp replica roughly halves wall clock and is still one judge instance. dp is derived from --gres,
# so the fallback is just a smaller allocation:
#   sbatch --gres=gpu:2 --mem=96G --cpus-per-task=8 --time=24:00:00 <this file>
# Both sizings give 48 GiB of host memory per GPU. Don't cut --mem: two replicas hold ~2x the
# pending tokens of one.
#
# No cross-job resume: the per-key skip only makes in-job retries cheap, and a cell re-judged in a
# second job lands on a different judge instance. If it walls: rm -rf $OUTROOT/j<deadjobid> and
# resubmit the whole job.
#
#   sbatch evaluation/reward_landscape/reaudit_all.slurm
#   DRY_RUN=1 / SMOKE=1 / ONLY=<substring> / N_GAMES=<n> for verification
#
# Optional env: RESULTS_ROOT MANIFEST OUTROOT TP_SIZE NGPU(--gres) DP_SIZE(gpus/tp)
#   MAX_WORKERS(160*dp) SGLANG_EXTRA JUDGE_PATH($MODELS_DIR/Qwen3.5-35B-A3B) JUDGE_NAME RL_VENV REPO
#   MODELS_DIR(<repo>/models) CACHE_DIR(<repo>/.cache) DRY_RUN(0) SMOKE(0)
#   SMOKE_ARM(only_anchoring) ONLY(substring on the key) N_GAMES
#   RL_MONITOR_JUDGE_RETRY(2): retry budget for the online and offline judges; don't also pass --retry.
# A DRY_RUN off a GPU node needs NGPU=<the allocation you will ask for>. sbatch --export drops
# empty assignments (`FOO=` never arrives), so blank a knob by giving it a real value.
set -uo pipefail

# Under sbatch the script runs from a spool copy, so REPO comes from SLURM_SUBMIT_DIR (submit from
# the repo root).
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
HERE="$REPO/evaluation/reward_landscape"
cd "$REPO" || exit 1
mkdir -p logs
RL_VENV="${RL_VENV:-$REPO/.venv}"; PY="$RL_VENV/bin/python"
[ -x "$PY" ] || { echo "[env] ERROR: no venv python at $PY (set RL_VENV)"; exit 1; }

RESULTS_ROOT="${RESULTS_ROOT:-$REPO/experiments/results}"
RESD="$RESULTS_ROOT/old-bailey"
MANIFEST="${MANIFEST:-$RESULTS_ROOT/reward_landscape/manifest.tsv}"
GATE_PY="$HERE/gate_audit.py"
[ -f "$GATE_PY" ] || { echo "[env] ERROR: gate_audit.py not found next to this script"; exit 1; }
FULL_TEMPLATE="$REPO/datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json"
TP_SIZE="${TP_SIZE:-2}"
# dp comes from the GPU count, not an env string: a `SGLANG_EXTRA=` meant to blank `--dp-size 2`
# is dropped by sbatch (see header), and tp*dp schedulers then fight over too few GPUs.
NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}}"
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
DP_SIZE="${DP_SIZE:-$(( NGPU / TP_SIZE ))}"
[ "$DP_SIZE" -ge 1 ] || DP_SIZE=1
[ $(( TP_SIZE * DP_SIZE )) -le "$NGPU" ] || {
    echo "[serve] ERROR: tp=$TP_SIZE x dp=$DP_SIZE needs $(( TP_SIZE * DP_SIZE )) GPUs, have $NGPU"; exit 1; }
DP_ARG=""; [ "$DP_SIZE" -gt 1 ] && DP_ARG="--dp-size $DP_SIZE"
# One replica saturates at ~141 concurrent requests; extra workers just queue and use host RAM.
MAX_WORKERS="${MAX_WORKERS:-$(( 160 * DP_SIZE ))}"
SGLANG_EXTRA="${SGLANG_EXTRA:-}"
JUDGE_NAME="${JUDGE_NAME:-qwen3.5-35B}"
JUDGE_PATH="${JUDGE_PATH:-${MODELS_DIR:-$REPO/models}/Qwen3.5-35B-A3B}"
OUTROOT="${OUTROOT:-$RESD/_reward_landscape_audit}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SMOKE_ARM="${SMOKE_ARM:-only_anchoring}"
# One parse-retry knob for both judges, so `metrics.evidence_audit_retry` in the fabrication
# sidecar matches the strategy driver's retry count. Passing --retry would override it.
export RL_MONITOR_JUDGE_RETRY="${RL_MONITOR_JUDGE_RETRY:-2}"
# The gate compares every artifact's judge_model against this name.
export JUDGE_NAME

OUTDIR="$OUTROOT/j${SLURM_JOB_ID:-0}"
BY="$OUTDIR/by_result"
FAILED_TSV="$OUTDIR/FAILED.tsv"

[ -s "$MANIFEST" ] || { echo "[env] ERROR: missing manifest $MANIFEST (run build_manifest.py)"; exit 1; }
[ -f "$FULL_TEMPLATE" ] || { echo "[env] ERROR: missing case template $FULL_TEMPLATE"; exit 1; }

# ---- build the per-key work list from the manifest -----------------------------------------------
# manifest (key sender profile arm result) -> key<TAB>result<TAB>sender<TAB>profile. sender is the
# results-directory slug, which audit_fabrications.py stamps as metrics.model.
WORK="$(mktemp)"; trap 'rm -f "$WORK"' EXIT
awk -F'\t' 'NR>1 && $1 != "" { printf "%s\t%s\t%s\t%s\n", $1, $5, $2, $3 }' "$MANIFEST" \
    | sort > "$WORK"

# ---- SMOKE: one arm across every sender and profile in the manifest -----------------------------
if [ "$SMOKE" = 1 ]; then
    _F="$(mktemp)"; grep -- "__${SMOKE_ARM}$(printf '\t')" "$WORK" > "$_F"
    [ -s "$_F" ] || { rm -f "$_F"; echo "[reaudit] ERROR: SMOKE_ARM='$SMOKE_ARM' selects no key in $MANIFEST"; exit 1; }
    mv "$_F" "$WORK"
    N_GAMES="${N_GAMES:-10}"
    echo "[reaudit] SMOKE=1 arm=$SMOKE_ARM -> $(wc -l < "$WORK") keys, N_GAMES=$N_GAMES"
fi
if [ -n "${ONLY:-}" ]; then
    _F="$(mktemp)"; grep -- "$ONLY" "$WORK" > "$_F"; mv "$_F" "$WORK"
    echo "[reaudit] ONLY=$ONLY -> $(wc -l < "$WORK") keys"
fi

N_KEYS=$(wc -l < "$WORK")
[ "$N_KEYS" -gt 0 ] || { echo "[reaudit] ERROR: no keys selected"; exit 1; }
echo "[reaudit] $N_KEYS keys, retry=$RL_MONITOR_JUDGE_RETRY, out=$OUTDIR"
echo "[reaudit] senders: $(cut -f3 "$WORK" | sort -u | tr '\n' ' ')"
echo "[reaudit] profiles: $(cut -f4 "$WORK" | sort -u | tr '\n' ' ')"

# ---- the two per-key invocations, in one place so DRY_RUN prints exactly what runs --------------
# A's --out must be byte-identical to B's --fabrication-json.
argv_fab() {   # $1=key $2=result $3=sender $4=profile
    printf '%s\0' "$PY" evaluation/audit_fabrications.py --domain old-bailey \
        --result-file "$2" --full-template "$FULL_TEMPLATE" \
        --model "$3" --receiver-config "$4" \
        --max-workers "$MAX_WORKERS" --no-wandb --out "$BY/$1.fabrication.json"
}
argv_strat() { # $1=key $2=result
    printf '%s\0' "$PY" -m rl.strategy_audit.run \
        --oldbailey-result "$2" \
        --out "$BY/$1.strategy_audit.json" \
        --fabrication-json "$BY/$1.fabrication.json" \
        --max-workers "$MAX_WORKERS"
    [ -n "${N_GAMES:-}" ] && printf '%s\0' --n-games "$N_GAMES"
    return 0
}
run_argv() { local -a a; mapfile -d '' -t a; "${a[@]}"; }
show_argv() { local -a a; mapfile -d '' -t a; printf '%q ' "${a[@]}"; printf '\n'; }

if [ "$DRY_RUN" = 1 ]; then
    echo "[reaudit] DRY_RUN: printing argv only, nothing executed"
    while IFS=$'\t' read -r K RES SENDER PROF; do
        argv_fab "$K" "$RES" "$SENDER" "$PROF" | show_argv
        argv_strat "$K" "$RES" | show_argv
    done < "$WORK"
    exit 0
fi

mkdir -p "$BY"
: > "$FAILED_TSV"
cp "$MANIFEST" "$OUTDIR/manifest.tsv"

note_fail() {  # pass, key, reason; appended at once so a walled job still leaves a record
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Is)" >> "$FAILED_TSV"
    echo "[reaudit] FAIL $1 $2: $3"
}

# A dead judge does not raise: receiver_client.chat swallows the connection error and returns '',
# so later cells "succeed" with ~100% parse failures and the gate discards them. Check liveness
# before each cell and abort once the server is gone; a restarted judge would be a new instance.
judge_alive() {
    kill -0 "${SGPID:-0}" 2>/dev/null || return 1
    curl --noproxy '*' -sf --max-time 20 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}
require_judge() {
    judge_alive && return 0
    echo "[reaudit] FATAL: the judge is gone (checked before $1). Serve-log tail:"
    tail -n 40 "$SGLOG"
    printf 'judge_died\t%s\t%s\t%s\n' "$1" "server unreachable" "$(date -Is)" >> "$FAILED_TSV"
    echo "[reaudit] aborting; rm -rf $OUTDIR and resubmit whole (no cross-job resume)"
    exit 2
}
on_term() {
    echo "[reaudit] SIGNAL received -- dumping progress"
    echo "[reaudit] fabrication=$(ls "$BY"/*.fabrication.json 2>/dev/null | wc -l)/$N_KEYS" \
         "strategy=$(ls "$BY"/*.strategy_audit.json 2>/dev/null | wc -l)/$N_KEYS"
    cat "$FAILED_TSV" 2>/dev/null
    kill "${SGPID:-0}" 2>/dev/null
    exit 143
}
trap on_term USR1 TERM

# --- network / caches / JIT toolchain -------------------------------------------------------------
module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
_CUDA_SYS="${CUDA_SYS:-/usr/local/cuda-13.0}"
_CUDA_INC="$RL_VENV/lib/python3.12/site-packages/nvidia/cu13/include"
export CUDA_HOME="${CUDA_HOME:-$CACHE_DIR/cuda13-merged}"
mkdir -p "$CUDA_HOME"
ln -sfn "$_CUDA_SYS/bin" "$CUDA_HOME/bin"; ln -sfn "$_CUDA_INC" "$CUDA_HOME/include"; ln -sfn "$_CUDA_SYS/lib64" "$CUDA_HOME/lib64"
export PATH="$RL_VENV/bin:$CUDA_HOME/bin:$PATH"; export CPATH="$_CUDA_INC:${CPATH:-}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$CACHE_DIR/tvm-ffi}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_DIR/flashinfer-base}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

unset PYTHONPATH || true
# Import checks before the ~3-minute model load, so a broken env fails in seconds.
"$PY" -c "import sglang" >/dev/null 2>&1 || { echo "[env] ERROR: cannot import sglang ($PY)"; exit 1; }
"$PY" -c "import rl.strategy_audit.run" >/dev/null 2>&1 || { echo "[env] ERROR: $PY lacks rl.strategy_audit"; exit 1; }
# The judge code must match what gate_audit.py and check_inputs.py require (a preset
# false_information column, the 1024/256 budget policy); otherwise the gate rejects cells hours in.
"$PY" - <<'PYEOF' || { echo "[env] ERROR: this checkout's audit code does not match the gate's contract"; exit 1; }
import sys, inspect
from rl.strategy_audit import run, audit
assert hasattr(run, "fi_indicator"), "run.fi_indicator missing"
assert "judge_budget_policy" in inspect.getsource(run._result_summary), "judge_budget_policy missing"
assert audit.judge_budget(True) == 1024 and audit.judge_budget(False) == 256, "judge budget policy wrong"
sys.exit(0)
PYEOF
"$PY" -m rl.strategy_audit.audit --selftest >/dev/null 2>&1 || { echo "[env] ERROR: audit --selftest failed"; exit 1; }
echo "[env] audit code matches the gate's contract and selftest passes"

# --- serve the 35B judge -------------------------------------------------------------------------
# The job-id port is only a starting guess: two jobs congruent mod 17000 can share a node and old
# sockets linger, so probe for a free port rather than lose the model load to a bind failure.
PORT=$(( 20000 + 2 * (${SLURM_JOB_ID:-0} % 17000) ))
for _off in $(seq 0 2 400); do
    _p=$(( PORT + _off ))
    if ! curl --noproxy '*' -sf --max-time 2 "http://127.0.0.1:${_p}/v1/models" >/dev/null 2>&1 \
       && ! ss -ltn 2>/dev/null | grep -q ":${_p} "; then PORT=$_p; break; fi
done
mkdir -p "$OUTDIR"
echo "[serve] using port $PORT"
SGLOG="logs/judge_reward_landscape_${SLURM_JOB_ID:-0}.log"
echo "[serve] $JUDGE_NAME <- $JUDGE_PATH on 127.0.0.1:$PORT"
echo "[serve] topology: ${NGPU} GPUs -> tp=$TP_SIZE dp=$DP_SIZE (${MAX_WORKERS} client workers)"
HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy= \
  "$PY" -m sglang.launch_server --model-path "$JUDGE_PATH" --served-model-name "$JUDGE_NAME" \
    --host 127.0.0.1 --port "$PORT" --tp-size "$TP_SIZE" --reasoning-parser qwen3 \
    ${DP_ARG} ${SGLANG_EXTRA:-} > "$SGLOG" 2>&1 &
SGPID=$!
trap 'kill "$SGPID" 2>/dev/null || true' EXIT
UP=0
for _i in $(seq 1 240); do
  kill -0 "$SGPID" 2>/dev/null || { echo "[serve] ERROR: judge died. Tail:"; tail -n 80 "$SGLOG"; exit 1; }
  curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { UP=1; echo "[serve] UP ($_i)"; break; }
  sleep 5
done
[ "$UP" = 1 ] || { echo "[serve] ERROR: judge not ready. Tail:"; tail -n 120 "$SGLOG"; exit 1; }
# audit_fabrications.py reads RECEIVER_MODEL_ID (not --judge-model) for --domain old-bailey, and
# the gate checks this judge name on every artifact.
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$PORT" RECEIVER_MODEL_ID="$JUDGE_NAME"
INSTANCE="job=${SLURM_JOB_ID:-0} node=$(hostname) date=$(date -Is) judge=$JUDGE_NAME path=$JUDGE_PATH gpus=$NGPU tp=$TP_SIZE dp=$DP_SIZE extra='${SGLANG_EXTRA}' workers=$MAX_WORKERS partition=${SLURM_JOB_PARTITION:-?}"
mkdir -p "$BY"
echo "$INSTANCE" > "$OUTDIR/_judge_instance.txt"
cp "$OUTDIR/_judge_instance.txt" "$BY/_judge_instance.txt"
echo "[reaudit] $INSTANCE"

GATE="$PY $GATE_PY"

# --- Pass A: fabrication for every result ---------------------------------------------------------
echo "[reaudit] ===== Pass A: fabrication, $N_KEYS keys ====="
A_OK=0
while IFS=$'\t' read -r K RES SENDER PROF; do
    FB="$BY/$K.fabrication.json"
    if [ -s "$FB" ] && $GATE --fabrication "$FB" --result "$RES" >/dev/null 2>&1; then
        A_OK=$(( A_OK + 1 )); continue                      # in-job retry only, never cross-job
    fi
    require_judge "fab:$K"
    echo "[fab] $K"
    if argv_fab "$K" "$RES" "$SENDER" "$PROF" | run_argv; then
        if $GATE --fabrication "$FB" --result "$RES"; then A_OK=$(( A_OK + 1 ))
        else note_fail fabrication "$K" "gate rejected"; rm -f "$FB"; fi
    else
        note_fail fabrication "$K" "driver exited nonzero"; rm -f "$FB"
    fi
done < "$WORK"
echo "[reaudit] Pass A: $A_OK/$N_KEYS"

# --- Pass B: the 42-way strategy audit -----------------------------------------------------------
# A missing fabrication sidecar is a hard failure: run.py would fall back to an in-process count,
# a different instrument (evidence from the result's own snapshot instead of the current full
# template, and a fresh 4096-token judge sample), so the key would not be comparable.
run_strategy() {  # $1=key $2=result
    local K="$1" RES="$2"
    local OUT="$BY/$K.strategy_audit.json"
    if [ ! -s "$BY/$K.fabrication.json" ]; then
        note_fail strategy "$K" "no fabrication sidecar -- would silently switch instrument"
        return 1
    fi
    if [ -s "$OUT" ] && $GATE --strategy "$OUT" --result "$RES" ${N_GAMES:+--n-games "$N_GAMES"} >/dev/null 2>&1; then
        return 0
    fi
    require_judge "strategy:$K"
    echo "[strategy] $K"
    if argv_strat "$K" "$RES" | run_argv; then
        if $GATE --strategy "$OUT" --result "$RES" ${N_GAMES:+--n-games "$N_GAMES"}; then return 0; fi
        note_fail strategy "$K" "gate rejected"; rm -f "$OUT"; return 1
    fi
    note_fail strategy "$K" "driver exited nonzero"; rm -f "$OUT"; return 1
}

echo "[reaudit] ===== Pass B: full 42-way, $N_KEYS keys ====="
B_OK=0
while IFS=$'\t' read -r K RES SENDER PROF; do
    run_strategy "$K" "$RES" && B_OK=$(( B_OK + 1 ))
done < "$WORK"
echo "[reaudit] Pass B: $B_OK/$N_KEYS"

# --- publish --------------------------------------------------------------------------------------
COMPLETE=0
if [ "$A_OK" = "$N_KEYS" ] && [ "$B_OK" = "$N_KEYS" ]; then COMPLETE=1; fi

echo "[reaudit] ===== summary ====="
echo "[reaudit] A=$A_OK/$N_KEYS  B=$B_OK/$N_KEYS"
if [ -s "$FAILED_TSV" ]; then
    echo "[reaudit] FAILED ($(wc -l < "$FAILED_TSV")):"; cat "$FAILED_TSV"
else
    echo "[reaudit] FAILED=[ none ]"
fi

if [ "$COMPLETE" != 1 ]; then
    echo "[reaudit] INCOMPLETE"
    exit 1
fi
if [ "$SMOKE" = 1 ] || [ -n "${ONLY:-}" ]; then
    # A subset run never publishes `final`; the analysis would read a partial grid.
    echo "[reaudit] SUBSET COMPLETE (smoke=$SMOKE only=${ONLY:-}) -- not published; artifacts in $BY"
    exit 0
fi
# `ln -sT` is required: plain `ln -s X final` on an existing symlink-to-dir dereferences it and
# silently creates final/X inside the published dir.
if ln -sT "j${SLURM_JOB_ID:-0}" "$OUTROOT/final" 2>/dev/null; then
    echo "[reaudit] PUBLISHED $OUTROOT/final -> j${SLURM_JOB_ID:-0}"
else
    echo "[reaudit] complete, but $OUTROOT/final already published by another racer (left unpublished)"
fi
echo "[reaudit] COMPLETE"
exit 0
