#!/bin/bash
# Driver for SFT-rollout generation (stages 1-2 of the SFT-then-RL curriculum). A teacher model
# plays the sender with the 42-strategy-guide prompt over the 200 SFT-train games
# (split_rl_sft.py --sft-holdout -> sft_split.json), PASSES temp-1 passes, against the Qwen3.5-35B
# receiver at RECEIVER_ENDPOINT=<node>:<port>. After each pass both per-turn audits run against the
# same served 35B and freeze their sidecars:
#   <res>.strategy_audit.json   (42-way, evidence-grounded, per-round vectors)
#   <res>.fabrication.json      (training rh_fake_evidence instrument, per-round)
# build_sft_dataset.py consumes result + sidecars and never re-judges. Audits and builder are
# teacher-agnostic; the builder's only model tie is the student Qwen tokenizer/format.
#
# SENDER_API picks the teacher transport:
#   gateway (default): a model hosted behind the API gateway (e.g. gpt-5.4-mini); run it on a
#     machine that can reach the gateway, with GATEWAY_URL and GATEWAY_API_KEY set. The
#     distillation path.
#   sglang: self-hosted Qwen at SENDER_ENDPOINT=<node>:<port>, --served-model-name must equal
#     SENDER. Self-distillation from any served checkpoint, fully local; see
#     collect_selfdistill.slurm, which co-serves sender + receiver and calls this driver.
#
# Diversity steering: passes >= STEER_FROM_PASS add --steer-strategies STEER_K --steer-seed <pass>
# (a per-game seeded "favor these allowed strategies" hint, generation only; the SFT prompt prefix
# is rebuilt from sft_holdout.parquet, so prompt parity is untouched).
#
# Resume: rollouts resume per pass file (--skip, keyed by game id). A pass whose audit sidecar
# exists and covers the pass is not re-audited (the judge samples at temp 1, so re-judging would
# change the survivor set); FORCE_AUDIT=1 re-runs audits.
#
# The two teachers the SFT inits are built with (submit from the repository root):
#   # gpt-5.4-mini distillation from a machine with API access and GATEWAY_URL and
#   # GATEWAY_API_KEY set; serve the 35B first and read its endpoint file:
#   sbatch --export=ALL,ENDPOINT_FILE=$PWD/logs/recv_endpoint.txt,SGLANG_VENV=$PWD/.venv \
#       scripts/serve_receiver_xnode.slurm
#   SENDER=gpt-5.4-mini SENDER_API=gateway RECEIVER_ENDPOINT=$(cat logs/recv_endpoint.txt) \
#       PROFILE=stubborn PROMPT=strategies PASSES=3 STEER_K=4 STEER_FROM_PASS=2 NUM_STEPS=3 \
#       bash datasets/old_bailey/sft/collect_rollouts.sh
#   # self-distillation: one GPU job co-serving the Qwen sender + the 35B, no key:
#   sbatch --export=ALL,SENDER=qwen3-8B-base,PASSES=10,STEER_K=0,BUILD_SFT=0 \
#       datasets/old_bailey/sft/collect_selfdistill.slurm
set -uo pipefail
SENDER="${SENDER:-gpt-5.4-mini}"
SENDER_API="${SENDER_API:-gateway}"    # gateway (hosted teacher) | sglang (served Qwen)
: "${RECEIVER_ENDPOINT:?set RECEIVER_ENDPOINT=<node>:<port> of the served 35B}"
RECEIVER="${RECEIVER:-qwen3.5-35B}"
PASSES="${PASSES:-3}"
PROFILE="${PROFILE:-stubborn}"         # receiver profile; must match the GRPO DIST (stubborn|neutral)
PROMPT="${PROMPT:-strategies}"         # sender prompt knob: strategies|base (must match the data root)
NUM_STEPS="${NUM_STEPS:-3}"
END_IDX="${END_IDX:-}"                 # cap after id selection; empty = all 200 SFT-train games
MAX_WORKERS="${MAX_WORKERS:-16}"
AUDIT_WORKERS="${AUDIT_WORKERS:-64}"
FAB_WORKERS="${FAB_WORKERS:-32}"
STEER_K="${STEER_K:-4}"                # legit slugs per steering hint (0 = never steer)
STEER_FROM_PASS="${STEER_FROM_PASS:-2}"  # first pass that steers (pass 1 unsteered = natural mix)
SKIP="${SKIP:-true}"
FORCE_AUDIT="${FORCE_AUDIT:-0}"

# REPO roots the venv/config/results (default: this checkout); pass REPO=/abs/path to relocate.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "[sft-driver] ERROR: python not found: $PY"; exit 1; }
"$PY" -c "import openai, pandas, requests, agents, sft.sft_dataset, rl.strategy_audit.run, rl.receiver_client" >/dev/null 2>&1 \
  || { echo "[sft-driver] ERROR: PY lacks openai/pandas/requests/agents/sft/rl.* (verl importable?)."; exit 1; }

# SFT-train ids and val membership come from the data root matching PROMPT (built by
# `python datasets/old_bailey/split_rl_sft.py --sft-holdout 200 --sender-prompt <spec> --out-dir <root>`).
case "$PROMPT" in
  strategies) _ROOT_DEFAULT="datasets/old_bailey/_generated/rl_sftsplit_strategies" ;;
  base)       _ROOT_DEFAULT="datasets/old_bailey/_generated/rl_sftsplit" ;;
  *) echo "[sft-driver] ERROR: unknown PROMPT '$PROMPT' (strategies|base)"; exit 1 ;;
esac
IDS_FILE="${IDS_FILE:-$_ROOT_DEFAULT/sft_split.json}"
# Val ids are identical across data roots and DISTs, so this root's bayesian parquet gives the
# held-out set to exclude, and no second root has to exist.
VAL_PARQUET="${VAL_PARQUET:-$_ROOT_DEFAULT/bayesian/rl_validation.parquet}"
FULL_TEMPLATE="${FULL_TEMPLATE:-datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json}"
[ -f "$IDS_FILE" ] || { echo "[sft-driver] ERROR: IDS_FILE missing: $IDS_FILE (run split_rl_sft.py --sft-holdout first)"; exit 1; }
[ -f "$VAL_PARQUET" ] || { echo "[sft-driver] ERROR: VAL_PARQUET missing: $VAL_PARQUET"; exit 1; }

RNODE="${RECEIVER_ENDPOINT%%:*}"; RPORT="${RECEIVER_ENDPOINT##*:}"
[ -n "$RNODE" ] && [ -n "$RPORT" ] || { echo "[sft-driver] ERROR: bad RECEIVER_ENDPOINT '$RECEIVER_ENDPOINT'"; exit 1; }

# Sender transport. sglang: self-hosted Qwen at SENDER_ENDPOINT, no key. gateway: requires
# GATEWAY_URL and GATEWAY_API_KEY (read at call time). Both transports then run with proxy
# variables unset.
SNODE=""; SPORT=""
if [ "$SENDER_API" = sglang ]; then
  : "${SENDER_ENDPOINT:?set SENDER_ENDPOINT=<node>:<port> of the served sender when SENDER_API=sglang}"
  SNODE="${SENDER_ENDPOINT%%:*}"; SPORT="${SENDER_ENDPOINT##*:}"
  [ -n "$SNODE" ] && [ -n "$SPORT" ] || { echo "[sft-driver] ERROR: bad SENDER_ENDPOINT '$SENDER_ENDPOINT'"; exit 1; }
elif [ "$SENDER_API" = gateway ]; then
  for _v in GATEWAY_URL GATEWAY_API_KEY; do
    if [ -z "${!_v:-}" ]; then
      eval "$(grep "^export ${_v}=" ~/.bashrc 2>/dev/null | head -1)" || true
    fi
    [ -n "${!_v:-}" ] || { echo "[sft-driver] ERROR: $_v unset (SENDER_API=gateway)."; exit 1; }
  done
else
  echo "[sft-driver] ERROR: unknown SENDER_API '$SENDER_API' (gateway|sglang)"; exit 1
fi
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
export NO_PROXY="localhost,127.0.0.1,${RNODE}${SNODE:+,$SNODE}"; export no_proxy="$NO_PROXY"

# Receiver + both audit judges all go through rl.receiver_client -> RECEIVER_HOST/PORT/MODEL_ID.
export RECEIVER_HOST="$RNODE" RECEIVER_PORT="$RPORT" RECEIVER_MODEL_ID="$RECEIVER"

got="$(curl --noproxy '*' -s "http://${RNODE}:${RPORT}/v1/models" \
        | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
[ "$got" = "$RECEIVER" ] || { echo "[sft-driver] ERROR: receiver not reachable at ${RNODE}:${RPORT} (got '$got')"; exit 1; }
echo "[sft-driver] receiver reachable: $got @ ${RNODE}:${RPORT}"

if [ "$SENDER_API" = sglang ]; then
  gots="$(curl --noproxy '*' -s "http://${SNODE}:${SPORT}/v1/models" \
          | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  [ "$gots" = "$SENDER" ] || { echo "[sft-driver] ERROR: sender not reachable / served-name mismatch at ${SNODE}:${SPORT} (got '$gots', want '$SENDER')"; exit 1; }
  echo "[sft-driver] sender reachable: $gots @ ${SNODE}:${SPORT}"
fi

# Sender transport args, fixed across passes (both defined in evaluation/rl_rollout.py; gateway
# posts to GATEWAY_URL).
if [ "$SENDER_API" = sglang ]; then
  SENDER_API_ARGS=(--sender-api sglang --sender-host "$SNODE" --sender-port "$SPORT" --sender-name "$SENDER")
else
  SENDER_API_ARGS=(--sender-api gateway --sender-name "$SENDER")
fi

SENDER_SLUG="${SENDER//\//-}"
RES_DIR="experiments/results/old-bailey/${SENDER_SLUG}/sft_rollouts"
mkdir -p "$RES_DIR"
SKIPFLAG=$([ "$SKIP" = true ] && echo "--skip" || echo "--no-skip")

echo "=== [sft-driver] sender=$SENDER api=$SENDER_API prompt=$PROMPT profile=$PROFILE passes=$PASSES steer=K${STEER_K}@pass>=${STEER_FROM_PASS} ids=$IDS_FILE end_idx=${END_IDX:-all} ==="
for (( p=1; p<=PASSES; p++ )); do
  RES="${RES_DIR}/${PROFILE}_sfttrain_${PROMPT}_pass${p}.json"
  STEER_ARGS=""
  if [ "$STEER_K" -gt 0 ] && [ "$p" -ge "$STEER_FROM_PASS" ]; then
    STEER_ARGS="--steer-strategies $STEER_K --steer-seed $p"
  fi
  echo "------------------------------------------------------------------"
  echo "[pass $p/$PASSES] -> $RES  (steer: ${STEER_ARGS:-off})"
  "$PY" evaluation/rl_rollout.py --receiver-api sglang --domain old-bailey --profile "$PROFILE" \
    "${SENDER_API_ARGS[@]}" \
    --sender-prompt "$PROMPT" \
    --split train --val-parquet "$VAL_PARQUET" --ids-file "$IDS_FILE" \
    --num-rounds "$NUM_STEPS" --max-workers "$MAX_WORKERS" \
    ${END_IDX:+--end-idx "$END_IDX"} $STEER_ARGS "$SKIPFLAG" --out "$RES" || exit 1
  [ -f "$RES" ] || { echo "[pass $p] ERROR: result not found: $RES"; exit 1; }

  # A sidecar is frozen only if it covers the pass (audited/row count >= the result's complete-game
  # count); one written before a resume extended the result (e.g. smoke END_IDX, then full) is stale.
  _covers() {  # _covers <sidecar> <count_key: n_games_audited|rows>
    "$PY" - "$RES" "$1" "$2" <<'PYEOF'
import json, sys
res, side, key = sys.argv[1], sys.argv[2], sys.argv[3]
n_complete = sum(1 for g in json.load(open(res)) if g.get("complete"))
d = json.load(open(side))
n_aud = len(d["rows"]) if key == "rows" else d.get("n_games_audited", 0)
sys.exit(0 if n_aud >= n_complete else 1)
PYEOF
  }
  AUD="${RES%.json}.strategy_audit.json"
  if [ -f "$AUD" ] && [ "$FORCE_AUDIT" != 1 ] && _covers "$AUD" n_games_audited; then
    echo "[pass $p] strategy audit sidecar exists and covers the pass (frozen), skipping: $AUD"
  else
    echo "[pass $p] per-turn strategy audit -> $AUD  (judge=$RECEIVER, 42 x rounds calls/game)"
    "$PY" -m rl.strategy_audit.run --oldbailey-result "$RES" --per-turn \
      --max-workers "$AUDIT_WORKERS" || exit 1
  fi
  FAB="${RES%.json}.fabrication.json"
  if [ -f "$FAB" ] && [ "$FORCE_AUDIT" != 1 ] && _covers "$FAB" rows; then
    echo "[pass $p] fabrication sidecar exists and covers the pass (frozen), skipping: $FAB"
  else
    echo "[pass $p] per-turn fabrication audit -> $FAB  (training instrument via rl.monitors)"
    "$PY" evaluation/audit_fabrications.py --domain old-bailey --result-file "$RES" \
      --full-template "$FULL_TEMPLATE" --per-turn --model "$SENDER" \
      --receiver-config "$PROFILE" --max-workers "$FAB_WORKERS" --no-wandb || exit 1
  fi
done
echo "[done] $PASSES passes complete. Next:"
# pass[0-9].json, not pass*.json, so the glob skips the per-pass .strategy_audit.json /
# .fabrication.json sidecars.
echo "  $PY $REPO/datasets/old_bailey/sft/build_sft_dataset.py --results ${RES_DIR}/${PROFILE}_sfttrain_${PROMPT}_pass[0-9].json \\"
echo "      --prompts-parquet ${_ROOT_DEFAULT}/${PROFILE/neutral/bayesian}/sft_holdout.parquet \\"
echo "      --out-dir datasets/old_bailey/_generated/sft/${PROMPT}_${PROFILE}"
