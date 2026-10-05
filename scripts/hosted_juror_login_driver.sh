#!/bin/bash
# Play the persuasion games against the hosted juror (DeepSeek-V4-Flash via the API gateway) from a
# machine with API access. It refuses to run inside a SLURM job. The roles are split: the sender is
# served by the SLURM job scripts/serve_sender_xnode.slurm, the juror is the API, and the audit judge
# is the qwen3.5-35B of scripts/serve_receiver_xnode.slurm, pinned via RECEIVER_MODEL_ID=qwen3.5-35B.
#
# Hosted-juror counterpart of scripts/{oldbailey,houseshowing,nutrition}_rl_eval.slurm; it writes the
# same result file name for the same (domain, profile, prompt, receiver).
#
#   # 1. serve the sender(s) as a SLURM job; CELLS_FILE is a tsv "cell<TAB>slug<TAB>ckpt<TAB>kind"
#   sbatch --export=ALL,CELLS_FILE=$PWD/cells.tsv,ENDPOINT_FILE=$PWD/logs/sender_ep.txt \
#          scripts/serve_sender_xnode.slurm
#   # 2. (for the audits) serve the 35B judge in a second job
#   sbatch --export=ALL,RECEIVER=qwen3.5-35B,ENDPOINT_FILE=$PWD/logs/judge_ep.txt \
#          scripts/serve_receiver_xnode.slurm
#   # 3. drive the games from a shell outside SLURM that can reach the API and both endpoints
#   ENDPOINT_FILE=$PWD/logs/sender_ep.txt RECEIVER_ENDPOINT="$(cat logs/judge_ep.txt)" \
#     STRATEGY_AUDIT=1 bash scripts/hosted_juror_login_driver.sh
#
# For each sender the serve job publishes: read <node>:<port>:<slug> from ENDPOINT_FILE, play the
# games, then write the slug to $ENDPOINT_FILE.done to release the next checkpoint. Each sender gets
# up to PASSES rollout passes with evaluation/rollout_integrity.py --repair in between, because a
# hosted juror call can fail and `--skip` replays exactly those games.
#
# Knobs (env):
#   DOMAIN(old-bailey|house-showing|nutrition)  PROFILE(stubborn|neutral)
#   PROMPT(base|strategies|single_strategy:<slug>)   RECEIVER(DeepSeek-V4-Flash)
#   ENDPOINT_FILE (required unless DRY_RUN=1)   RECEIVER_ENDPOINT(<node>:<port> of the 35B judge)
#   SPLIT(val; old-bailey only)  END_IDX(all games in scope)  NUM_STEPS(3)  MAX_WORKERS(16)
#   PASSES(4)  RES_DIR(<repo>/experiments/results)  VAL_PARQUET(old-bailey held-out ids)
#   AUDIT_FAB(1)  STRATEGY_AUDIT(0)  DRY_RUN(0)  PY(<repo>/.venv/bin/python)
#   EP_WAIT_TRIES(4320) / EP_WAIT_NEXT(240): 5-second polls waiting for the first / the next sender.
#
# DRY_RUN=1 prints every command and the result stem, with no network call, key, gateway URL or
# endpoint file. GATEWAY_URL (the gateway's chat-completions URL) and GATEWAY_API_KEY are read from
# the environment; the key is never printed.
set -euo pipefail

[ -z "${SLURM_JOB_ID:-}" ] || {
  echo "ERROR: this driver runs outside SLURM (it is inside job $SLURM_JOB_ID), on a machine that"
  echo "       can reach the API gateway. Serve the sender as a job instead"
  echo "       (scripts/serve_sender_xnode.slurm) and run this from a login shell."
  exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-$REPO/.venv/bin/python}"

DOMAIN="${DOMAIN:-old-bailey}"
PROFILE="${PROFILE:-stubborn}"
PROMPT="${PROMPT:-base}"
RECEIVER="${RECEIVER:-DeepSeek-V4-Flash}"
SPLIT="${SPLIT:-val}"
END_IDX="${END_IDX:-}"
NUM_STEPS="${NUM_STEPS:-3}"
MAX_WORKERS="${MAX_WORKERS:-16}"
PASSES="${PASSES:-4}"
# Results root, laid out like the SLURM launchers: $RES_DIR/<domain>/<sender slug>/<stem>.json
RES_DIR="${RES_DIR:-$REPO/experiments/results}"
VAL_PARQUET="${VAL_PARQUET:-$REPO/datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet}"
AUDIT_FAB="${AUDIT_FAB:-1}"
STRATEGY_AUDIT="${STRATEGY_AUDIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
RECEIVER_ENDPOINT="${RECEIVER_ENDPOINT:-}"
EP_WAIT_TRIES="${EP_WAIT_TRIES:-4320}"   # x5s = 6 h, so a queued serve job is never called dead
EP_WAIT_NEXT="${EP_WAIT_NEXT:-240}"      # x5s = 20 min between one sender and the next

[[ -x "$PY" ]] || { echo "ERROR: python not found at '$PY' (set PY)."; exit 1; }

# Domain table: result-dir name, the stem's domain token, the metrics stage and its ground truth.
case "$DOMAIN" in
  old-bailey)
    DOM_DIR="old-bailey"; STEM_DOMAIN="oldbailey"
    EVALUATOR="evaluation/old_bailey/evaluate_oldbailey.py"
    FULL_TEMPLATE="datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json" ;;
  house-showing)
    DOM_DIR="house-showing"; STEM_DOMAIN="house_showing"
    EVALUATOR="evaluation/house_showing/evaluate_houseshowing.py"
    FULL_TEMPLATE="datasets/house_showing/processed/full/house_showing_full.json" ;;
  nutrition)
    DOM_DIR="nutrition"; STEM_DOMAIN="nutrition"
    EVALUATOR="evaluation/nutrition/evaluate_nutrition.py"
    FULL_TEMPLATE="datasets/nutrition/processed/full/nutrition_full.json" ;;
  *) echo "ERROR: DOMAIN must be old-bailey | house-showing | nutrition (got '$DOMAIN')"; exit 1 ;;
esac
# The juror's stated prior P(guilty), as in the launchers' profile_prior.
case "$PROFILE" in
  stubborn) PRIOR=0.1 ;;
  neutral)  PRIOR=0.5 ;;
  *) echo "ERROR: PROFILE must be stubborn | neutral (got '$PROFILE')"; exit 1 ;;
esac
# Old Bailey is the only domain with an RL split; the other two play their whole template.
if [ "$DOMAIN" != old-bailey ] && [ "$SPLIT" != val ]; then
  echo "[hosted-juror] note: SPLIT is old-bailey-only; ignoring SPLIT=$SPLIT for $DOMAIN"
fi

# Hosted jurors only; a served juror belongs in the SLURM launchers, which co-serve it.
_RECV_TRANSPORT="$("$PY" -m evaluation.receiver_models --receiver "$RECEIVER" --print-transport)" \
  && _RECV_RC=0 || _RECV_RC=$?
[ "${_RECV_TRANSPORT}:${_RECV_RC}" = "gateway:0" ] || {
  echo "ERROR: RECEIVER=$RECEIVER is not a hosted juror (transport '${_RECV_TRANSPORT}', exit $_RECV_RC)."
  echo "       This driver exists for the API juror; a locally served one is played by"
  echo "       scripts/{oldbailey,houseshowing,nutrition}_rl_eval.slurm, which serve it themselves."
  echo "       Configured jurors: $("$PY" -m evaluation.receiver_models --list | cut -f1 | tr '\n' ' ')"
  exit 1; }

# The key is only tested for presence, never printed. No `set -x` in this file: it would leak the
# key into the log.
if [ "$DRY_RUN" != 1 ] && [ -z "${GATEWAY_API_KEY:-}" ]; then
  echo "ERROR: GATEWAY_API_KEY is unset or empty, so the hosted juror cannot authenticate."
  echo "       Export it before running (if ~/.bashrc guards it behind an interactive-shell"
  echo "       check, a batch/driver shell needs:"
  echo "       eval \"\$(grep -m1 '^export GATEWAY_API_KEY=' ~/.bashrc)\")."
  exit 1
fi
if [ "$DRY_RUN" != 1 ] && [ -z "${GATEWAY_URL:-}" ]; then
  echo "ERROR: GATEWAY_URL is unset or empty, so the hosted juror has no endpoint."
  echo "       Export the gateway's chat-completions URL before running."
  exit 1
fi

# Pin the audit judge (a separate rl.receiver_client process that never follows the juror), so a
# stray judge call fails loudly against a dead endpoint instead of being answered by the gateway.
export RECEIVER_MODEL_ID=qwen3.5-35B
# Stock sampling: _hosted_receiver_sampling forwards only temperature/top_p/max_tokens, so these two
# knobs are the only way two runs could sample differently.
unset RL_RECV_TEMPERATURE RL_RECV_TOP_P
RNODE=""
if [ -n "$RECEIVER_ENDPOINT" ]; then
  RNODE="${RECEIVER_ENDPOINT%%:*}"
  export RECEIVER_HOST="$RNODE" RECEIVER_PORT="${RECEIVER_ENDPOINT##*:}"
fi
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# Result stem as the launchers build it: no prompt token for base, "_sender-<result stem>" otherwise.
"$PY" -m rl.sender_prompts --check "$PROMPT" >/dev/null || {
  echo "ERROR: unknown PROMPT '$PROMPT' (base | strategies | single_strategy:<slug>)"; exit 1; }
RES_TAG=""
if [ "$PROMPT" != base ]; then
  RES_TAG="_sender-$("$PY" -m rl.sender_prompts --print result-stem --domain "$DOMAIN" --prompt "$PROMPT")"
fi
RECV_SLUG="${RECEIVER//\//-}"
if [ "$DOMAIN" = old-bailey ]; then
  STEM="${PROFILE}_${STEM_DOMAIN}_rlrollout_${SPLIT}${RES_TAG}__recv_${RECV_SLUG}"
else
  STEM="${PROFILE}_${STEM_DOMAIN}_rlrollout${RES_TAG}__recv_${RECV_SLUG}"
fi

echo "=== [hosted-juror] domain=$DOMAIN profile=$PROFILE prompt=$PROMPT receiver=$RECEIVER ==="
echo "[hosted-juror] result stem: ${STEM}.json  (under $RES_DIR/$DOM_DIR/<sender slug>/)"
echo "[hosted-juror] audit judge: $RECEIVER_MODEL_ID @ ${RECEIVER_ENDPOINT:-<no RECEIVER_ENDPOINT>}"

VAL_ARG=()
if [ "$DOMAIN" = old-bailey ] && [ "$SPLIT" != all ] && [ -n "$VAL_PARQUET" ]; then
  if [ -f "$VAL_PARQUET" ]; then
    VAL_ARG=(--val-parquet "$VAL_PARQUET")
  else
    echo "[hosted-juror] WARN: VAL_PARQUET not found ($VAL_PARQUET); falling back to the seed-42 shuffle"
  fi
fi

FAILED=0

run_cmd() {   # print, and run unless this is a dry run
  printf '  '; printf '%q ' "$@"; printf '\n'
  [ "$DRY_RUN" = 1 ] || "$@"
}

judge_cmd() { # print, and run only when a judge endpoint is live and this is not a dry run
  printf '  '; printf '%q ' "$@"; printf '\n'
  { [ "$DRY_RUN" = 1 ] || [ -z "$RECEIVER_ENDPOINT" ]; } || "$@"
}

stage_warn() {  # $1 = stage name, $2 = sender slug
  # Count the failure and keep going, so it costs one artifact, not the remaining checkpoints.
  echo "[hosted-juror] WARN: the $1 stage failed for $2; rerun it on this result file"
  FAILED=$((FAILED + 1))
}

play_sender() {   # $1 = sender node, $2 = sender port, $3 = served-model slug
  local SNODE="$1" SPORT="$2" SLUG="$3"
  local OUT_DIR="$RES_DIR/$DOM_DIR/${SLUG//\//-}"
  local RES="$OUT_DIR/${STEM}.json"
  local pass complete=0

  export NO_PROXY="localhost,127.0.0.1,::1,$SNODE${RNODE:+,$RNODE}"
  export no_proxy="$NO_PROXY"
  echo "[hosted-juror] ==== $SLUG @ ${SNODE}:${SPORT} -> $RES"
  [ "$DRY_RUN" = 1 ] || mkdir -p "$OUT_DIR"

  local ROLL=("$PY" evaluation/rl_rollout.py
              --receiver "$RECEIVER"
              --domain "$DOMAIN" --profile "$PROFILE"
              --sender-host "$SNODE" --sender-port "$SPORT" --sender-name "$SLUG"
              --sender-prompt "$PROMPT"
              --num-rounds "$NUM_STEPS" --max-workers "$MAX_WORKERS" --skip --out "$RES")
  if [ "$DOMAIN" = old-bailey ]; then ROLL+=(--split "$SPLIT" "${VAL_ARG[@]}"); fi
  if [ -n "$END_IDX" ]; then ROLL+=(--end-idx "$END_IDX"); fi

  for pass in $(seq 1 "$PASSES"); do
    echo "[hosted-juror] rollout pass $pass/$PASSES"
    run_cmd "${ROLL[@]}" || echo "[hosted-juror] pass $pass ended early; the next pass resumes it"
    # Demote games with an empty juror/sender turn so --skip replays them. Both calls take
    # $NUM_STEPS, since "short" (damaged) is measured against it.
    run_cmd "$PY" evaluation/rollout_integrity.py --repair "$RES" --rounds "$NUM_STEPS" || true
    if [ "$DRY_RUN" = 1 ]; then complete=1; break; fi
    if "$PY" evaluation/rollout_integrity.py --check "$RES" --rounds "$NUM_STEPS" >/dev/null 2>&1; then
      complete=1; echo "[hosted-juror] $SLUG complete after pass $pass"; break
    fi
  done

  if [ "$complete" != 1 ]; then
    echo "[hosted-juror] WARN: $RES is still incomplete after $PASSES pass(es); no metrics for it"
    FAILED=$((FAILED + 1))
    return 0
  fi

  # Check who played from the artifact, not the file name, so a juror that silently fell back is
  # never reported as the hosted one. Fatal on purpose.
  echo "[hosted-juror] juror check: every game must carry receiver_model = gateway:$RECEIVER"
  run_cmd "$PY" -c 'import json, sys
want, path = sys.argv[1], sys.argv[2]
games = json.load(open(path))
bad = [g.get("id") for g in games if g.get("receiver_model") != want]
if bad:
    sys.exit(f"[hosted-juror] ERROR: {len(bad)} of {len(games)} game(s) were not played by {want} "
             f"(first ids: {bad[:5]})")
print(f"[hosted-juror] juror OK: all {len(games)} game(s) carry receiver_model={want}")' \
    "gateway:$RECEIVER" "$RES"

  if [ "$DOMAIN" = old-bailey ]; then
    # Pure python: the belief metrics read the juror's own replies, no judge call.
    echo "[hosted-juror] persuasion metrics (prior=$PRIOR) -> ${RES}.metrics.json"
    run_cmd "$PY" "$EVALUATOR" --result-file "$RES" --model "$SLUG" \
      --receiver-config "$PROFILE" --prior "$PRIOR" --rounds "$NUM_STEPS" --no-wandb \
      || stage_warn "persuasion metrics" "$SLUG"
  else
    # The OOD evaluators are their game's deception audit, so they need the served judge and must
    # run before the strategy audit, which reads their sidecar.
    echo "[hosted-juror] deception metrics (judge $RECEIVER_MODEL_ID) -> ${RES%.json}.deception.json"
    judge_cmd "$PY" "$EVALUATOR" --result-file "$RES" --full-template "$FULL_TEMPLATE" \
      --model "$SLUG" --receiver-config "$PROFILE" --max-workers "$MAX_WORKERS" --no-wandb \
      || stage_warn "deception metrics" "$SLUG"
  fi

  # Fabrication audit first: the strategy audit reads its sidecar for false_information.
  if [ "$DOMAIN" = old-bailey ] && [ "$AUDIT_FAB" = 1 ]; then
    echo "[hosted-juror] fabrication audit (judge $RECEIVER_MODEL_ID) -> ${RES%.json}.fabrication.json"
    judge_cmd "$PY" evaluation/audit_fabrications.py --domain old-bailey --result-file "$RES" \
      --full-template "$FULL_TEMPLATE" --model "$SLUG" --receiver-config "$PROFILE" \
      --max-workers "$MAX_WORKERS" --no-wandb \
      || stage_warn "fabrication audit" "$SLUG"
  fi
  if [ "$STRATEGY_AUDIT" = 1 ]; then
    echo "[hosted-juror] 42-way strategy audit (judge $RECEIVER_MODEL_ID) -> ${RES%.json}.strategy_audit.json"
    judge_cmd "$PY" -m rl.strategy_audit.run --result "$RES" --domain "$DOMAIN" \
      --max-workers "$MAX_WORKERS" \
      || stage_warn "strategy audit" "$SLUG"
  fi
  # Judge stages exist for every OOD game and for an Old Bailey game that asked for an audit.
  if [ -z "$RECEIVER_ENDPOINT" ] \
     && { [ "$DOMAIN" != old-bailey ] || [ "$AUDIT_FAB" = 1 ] || [ "$STRATEGY_AUDIT" = 1 ]; }; then
    echo "[hosted-juror] RECEIVER_ENDPOINT is unset, so the judge stage(s) above were only printed."
    echo "               Serve the 35B judge, then re-run them with its endpoint:"
    echo "  sbatch --export=ALL,RECEIVER=qwen3.5-35B,ENDPOINT_FILE=$REPO/logs/judge_ep.txt scripts/serve_receiver_xnode.slurm"
    echo "  RECEIVER_MODEL_ID=qwen3.5-35B RECEIVER_HOST=<node> RECEIVER_PORT=<port>  # from that file"
  fi
}

if [ "$DRY_RUN" = 1 ]; then
  echo "[hosted-juror] DRY_RUN=1: printing the commands for one sender, nothing is run"
  play_sender "<sender-node>" "<sender-port>" "<sender-slug>"
  exit 0
fi

: "${ENDPOINT_FILE:?set ENDPOINT_FILE (the path passed to scripts/serve_sender_xnode.slurm)}"
DONE_FILE="${ENDPOINT_FILE}.done"
declare -A HANDLED=()
n_senders=0
echo "[hosted-juror] waiting for $ENDPOINT_FILE ..."
while true; do
  # Wait for an endpoint not yet played. The serve job republishes the file per checkpoint, so no
  # new endpoint within the wait means it has finished.
  ep=""
  tries="$EP_WAIT_TRIES"; [ "$n_senders" -gt 0 ] && tries="$EP_WAIT_NEXT"
  for _i in $(seq 1 "$tries"); do
    if [ -f "$ENDPOINT_FILE" ]; then
      ep="$(cat "$ENDPOINT_FILE" 2>/dev/null || true)"
      [ -n "$ep" ] && [ -z "${HANDLED[$ep]:-}" ] && break
    fi
    ep=""; sleep 5
  done
  [ -n "$ep" ] || { echo "[hosted-juror] no new endpoint within the wait; the serve job is finished"; break; }

  SNODE="${ep%%:*}"; rest="${ep#*:}"; SPORT="${rest%%:*}"; SLUG="${rest#*:}"
  play_sender "$SNODE" "$SPORT" "$SLUG"
  HANDLED[$ep]=1
  n_senders=$((n_senders + 1))
  # Ack: the serve job advances to the next checkpoint when it sees this slug.
  printf '%s\n' "$SLUG" > "$DONE_FILE"
done

echo "[hosted-juror] done: $n_senders sender(s) played, $FAILED incomplete or failed stage(s)"
exit $(( FAILED > 0 ? 1 : 0 ))
