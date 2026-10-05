#!/bin/bash
#SBATCH --job-name=ob_rl_eval
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Old Bailey in-domain RL-checkpoint eval (sibling of houseshowing_rl_eval.slurm). An RL-trained
# sender (or base policy) plays the prosecutor against a fixed Qwen3.5-35B juror on the held-out
# games (default SPLIT=val: the 100 games of the RL seed-42 split the sender never trained on).
# Two scoring stages run per profile:
#   1. evaluation/old_bailey/evaluate_oldbailey.py: persuasion metrics (final P(guilty), margin over
#      the prior, belief shift, per-round series) with a profile-matched --prior.
#   2. evaluation/audit_fabrications.py --domain old-bailey: the training fabrication instrument
#      (rl/monitors._audit_evidence, judged by the co-served 35B via rl.receiver_client, as in GRPO)
#      -> <result>.fabrication.json.
#
# Co-serves two SGLang models on disjoint GPUs: the sender (TP=1, GPU 0) and the 35B receiver/judge
# (TP=2, GPU 1-2). The game stage is evaluation/rl_rollout.py, the RL-parity driver that replays the
# verl GRPO conversation (multi-turn sender chat, <receiver_response> turns, training receiver
# prompt/sampling via rl.receiver_client). Native <think> is off everywhere (driver sends
# enable_thinking=false; both servers run --reasoning-parser qwen3).
#
# Profiles: neutral (prior 0.5) and stubborn (0.1; the same rl/game_rows prose as the GRPO parquets).
# wandb is off; metrics land in the sidecar JSONs.
#
# Submit one sender (the 35B needs an 80 GB card):
#   sbatch --export=ALL,SENDER=qwen3-8B-rl-stubborn-j35B-fakepen0 scripts/oldbailey_rl_eval.slurm
#   # smoke (2 games, 1 profile):
#   sbatch --export=ALL,SENDER=qwen3-8B-base,END_IDX=2,PROFILES=neutral,MAX_WORKERS=2 scripts/oldbailey_rl_eval.slurm
#
# Required: SENDER (a model key in resolve_model below, or pass SENDER_MODEL_PATH).
# Optional: RECEIVER(qwen3.5-35B; served by this job; play a hosted juror with
#           scripts/hosted_juror_login_driver.sh)
#           PROFILES("neutral stubborn") SPLIT(val|train|all)
#           PROMPT(base|strategies|single_strategy:<slug>; the sender prompt, rl/sender_prompts.py)
#           VAL_PARQUET(datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet; empty =
#           seed-42 shuffle)
#           NUM_STEPS(3) END_IDX(100) MAX_WORKERS(16) SENDER_TP(1) RECEIVER_TP(2)
#           SENDER_MODEL_PATH/RECEIVER_MODEL_PATH/SGLANG_EXTRA
#           MERGED(<repo>/experiments/results/rl/_em_merged)
#           SGLANG_VENV(<repo>/.venv) INFER_PYTHON
#           MODELS_DIR(<repo>/models; one directory per base model) CACHE_DIR(<repo>/.cache)
#           IDS_FILE/IDS_KEY(game-id restriction)  RES_TAG(filename tag before "__recv_")
#           AUDIT_FAB(1; 0 skips the in-job fabrication audit)
set -euo pipefail

: "${SENDER:?Must set SENDER (a sender model key; see resolve_model below)}"
RECEIVER="${RECEIVER:-qwen3.5-35B}"
PROFILES="${PROFILES:-neutral stubborn}"
SPLIT="${SPLIT:-val}"
# Held-out ids come from a training parquet's extra_info.index, so they cannot drift from training.
# They match across DISTs, so the bayesian parquet is a safe default (use the run's own for
# non-default builds); empty = seed-42 shuffle.
VAL_PARQUET="${VAL_PARQUET:-datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet}"
NUM_STEPS="${NUM_STEPS:-3}"
END_IDX="${END_IDX:-100}"
MAX_WORKERS="${MAX_WORKERS:-16}"
# Sender prompt (rl/sender_prompts.py): base | strategies (the 42-technique ALLOWED/FORBIDDEN guide)
# | single_strategy:<slug>. Use the prompt the checkpoint was trained on; a mismatch measures the
# prompt, not the checkpoint. rl_rollout.py records it on every game.
PROMPT="${PROMPT:-base}"
# Reject SENDER_CONFIG rather than ignore it and run the base prompt under a guide-looking command.
[ -z "${SENDER_CONFIG:-}" ] || { echo "ERROR: the sender prompt is selected with PROMPT=base|strategies|single_strategy:<slug> (got SENDER_CONFIG='$SENDER_CONFIG')"; exit 1; }
# Game-id restriction for rl_rollout.py --ids-file/--ids-key, applied after the split selection.
# E.g. IDS_FILE=datasets/old_bailey/_generated/rl_sftsplit_strategies/sft_split.json with SPLIT=train
# keeps the 921 GRPO-draw games (drops the 200 SFT-holdout games).
IDS_FILE="${IDS_FILE:-}"
IDS_KEY="${IDS_KEY:-rl_train_ids}"
# Filename tag inserted before "__recv_" so a non-base PROMPT gets its own result file instead of
# sharing the base-prompt name of the same SENDER key. Empty for base. Use the stem
# evaluation/reward_landscape/rollout_sweep.slurm uses:
#   RES_TAG=_sender-$(python -m rl.sender_prompts --print result-stem --prompt strategies)
RES_TAG="${RES_TAG:-}"
# AUDIT_FAB=0 skips the in-job (game-level) fabrication audit, for collection runs labelled later in
# one separate judge job. A study needing --per-turn verdicts would otherwise pay the 35B twice and
# split its labels across two serve instances, whose aggregate rates differ (one-instance audit rule).
AUDIT_FAB="${AUDIT_FAB:-1}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
SGLANG_VENV="${SGLANG_VENV:-$REPO/.venv}"
SGLANG_PY="${SGLANG_PY:-$SGLANG_VENV/bin/python}"

# --- juror transport guard (config/receiver/model/, evaluation/receiver_models.py) ---
# RECEIVER is also the served-model name, audit-judge id and "__recv_" stem, so reject a hosted
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
# SFT checkpoint root (plain HF dirs, no merge step).
REPO_SFT="${REPO_SFT:-$REPO/experiments/results/sft}"
resolve_model() {   # $1 = model key -> sets RM_PATH, RM_TP
  # Stubborn fakepen0: the canonical runs are the _argfix ones. Merge a missing dir with
  # scripts/merge_verl_ckpt.slurm from the run's global_step_<N>/actor (RL_SRC = root of the run dirs).
  case "$1" in
    qwen3-4B-rl-stubborn-j35B-fakepen0)   RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_argfix_gs100";  RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0)   RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_argfix_gs100";  RM_TP=1 ;;
    # fmt0p1 pair (turn-level format bonus) at gs50 and gs100, for the step-0/50/100 trajectory.
    # Base-prompt checkpoints: evaluate with PROMPT=base.
    qwen3-4B-rl-stubborn-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-4B-rl-stubborn-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    # Trained with PROMPT=strategies; evaluate with it too. The _gs<step> suffix must match the
    # merged dir (scripts/merge_verl_ckpt.slurm; default 100 steps -> _gs100).
    qwen3-4B-rl-stubborn-j35B-fakepen0-strategies) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-j35B-fakepen0-strategies) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_gs100"; RM_TP=1 ;;
    # SFT-then-RL distill (921-game sftsplit data): from-base and SFT-initialized arms, trained
    # against the grounded-misrepresentation-v2 prompt/audit (misrepv2). Pair with PROMPT=strategies.
    qwen3-4B-rl-stubborn-sftsplit-misrepv2)         RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-misrepv2)         RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs100"; RM_TP=1 ;;
    # gs50 points of the two from-base arms (merged via ONLY=misrepv2_gs50). Pair with PROMPT=strategies.
    qwen3-4B-rl-stubborn-sftsplit-misrepv2-gs50)    RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs50"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-misrepv2-gs50)    RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs50"; RM_TP=1 ;;
    qwen3-4B-rl-stubborn-sftsplit-misrepv2-sftinit) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-misrepv2-sftinit) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2_gs100"; RM_TP=1 ;;
    # Self-distillation arms (teacher == student, PASSES=10), SFT-initialized from the self-distilled
    # demonstrations. Pair with PROMPT=strategies.
    qwen3-4B-rl-stubborn-sftsplit-selfdistill) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-selfdistill) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2_gs100"; RM_TP=1 ;;
    # LR ablation (4B from-base, lr 3e-6 vs default 1e-6): gs100 matches the 100-step main arms;
    # gs200 is its full run.
    qwen3-4B-rl-stubborn-sftsplit-misrepv2-lr3e-6)      RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6_gs100"; RM_TP=1 ;;
    qwen3-4B-rl-stubborn-sftsplit-misrepv2-lr3e-6-gs200) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6_gs200"; RM_TP=1 ;;
    # illegal_full11 arms: lambda=0.3 over 11 additive binary illegal-strategy penalties (vs the
    # 2-term belief+fab pen0p3 keys), same 921-game sftsplit_strategies data / stubborn juror /
    # fmt0p1, x 3 GRPO inits. Pair with PROMPT=strategies (the parquets bake the 42-strategy guide)
    # and STRATEGY_AUDIT=1. One glob covers every merged step (gs100/150/200):
    #   qwen3-<4B|8B>-rl-stubborn-sftsplit-<base|misrepv2-sftinit|selfdistill-sftinit>-ill11-gs<step>
    qwen3-*B-rl-stubborn-sftsplit-*-ill11-gs*)
      _sz="${1#qwen3-}";  _sz="${_sz%%-*}"                  # 4B | 8B
      _gs="${1##*-ill11-gs}"                                # 100 | 150 | 200
      _arm="${1#qwen3-${_sz}-rl-stubborn-sftsplit-}"; _arm="${_arm%-ill11-gs${_gs}}"
      case "$_arm" in
        base)                _suf="_misrepv2" ;;
        misrepv2-sftinit)    _suf="_misrepv2_sftinit-ep2" ;;
        selfdistill-sftinit) _suf="_sftinit-selfdistill-ep2" ;;
        # gpt-5.4-mini distillation plus one epoch of base_origin probe SFT (two SFT stages).
        auxsft647-sftinit)   _suf="_misrepv2_sftinit-ep2auxce647" ;;
        *) # Unknown arm on an ill11-shaped key: a typo, so fail with the valid arms, unless an
           # explicit *_MODEL_PATH override means the key never has to resolve (see the call site).
           if [ "${_ALLOW_UNKNOWN:-0}" = 1 ]; then
             # The common branch (in-repo launchers pass explicit paths). Warn, since a misspelled
             # arm still runs, under a result dir named for the typo.
             echo "[serve] NOTE: model key '$1' names an unregistered init arm '$_arm'; using the" >&2
             echo "[serve]       explicit *_MODEL_PATH override. Check the arm spelling if the" >&2
             echo "[serve]       result lands in an unexpected directory." >&2
             RM_PATH=""; RM_TP=1; return 0
           fi
           echo "[serve] ERROR: unknown illegal_full11 init arm '$_arm' in model key '$1'." >&2
           echo "[serve]        expected base | misrepv2-sftinit | selfdistill-sftinit |" >&2
           echo "[serve]        auxsft647-sftinit, as" >&2
           echo "[serve]        qwen3-<4B|8B>-rl-stubborn-sftsplit-<arm>-ill11-gs<step>," >&2
           echo "[serve]        or pass SENDER_MODEL_PATH / RECEIVER_MODEL_PATH (whichever role" >&2
           echo "[serve]        names this key) to bypass path resolution." >&2
           exit 1 ;;
      esac
      RM_PATH="$MERGED/sender_qwen3-${_sz}_stubborn_j35B_pen0p3_illegal_full11_fmt0p1_strategies_sftsplit${_suf}_gs${_gs}"
      RM_TP=1 ;;
    # Prompt-parity keys for the self-distill column: the same weights under the 7898-char
    # 42-strategy guide they trained with (PROMPT=strategies) instead of the 682-char base prompt.
    # A distinct SENDER key gives a distinct result dir, so the two prompts never share a file.
    qwen3-4B-rl-stubborn-sftsplit-selfdistill-sftinit-pen0p3-gs100-strat) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_sftinit-selfdistill-ep2_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-selfdistill-sftinit-pen0p3-gs100-strat) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_sftinit-selfdistill-ep2_gs100"; RM_TP=1 ;;
    qwen3-4B-sft-strategies-stubborn-selfdistill-strat)              RM_PATH="$REPO_SFT/sft_qwen3-4B_strategies_stubborn_selfdistill_ep2/hf_final"; RM_TP=1 ;;
    qwen3-8B-sft-strategies-stubborn-selfdistill-strat)              RM_PATH="$REPO_SFT/sft_qwen3-8B_strategies_stubborn_selfdistill_ep2/hf_final"; RM_TP=1 ;;
    # The same -strat keys for the from-base column at gs150/gs200.
    qwen3-4B-rl-stubborn-sftsplit-base-pen0p3-gs150-strat) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_gs150"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-base-pen0p3-gs150-strat) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_gs150"; RM_TP=1 ;;
    qwen3-4B-rl-stubborn-sftsplit-base-pen0p3-gs200-strat) RM_PATH="$MERGED/sender_qwen3-4B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_gs200"; RM_TP=1 ;;
    qwen3-8B-rl-stubborn-sftsplit-base-pen0p3-gs200-strat) RM_PATH="$MERGED/sender_qwen3-8B_stubborn_j35B_pen0p3_belief_fab_fmt0p1_strategies_sftsplit_gs200"; RM_TP=1 ;;
    qwen3-14B-rl-stubborn-j35B-fakepen0)  RM_PATH="$MERGED/sender_qwen3-14B_stubborn_j35B_fakepen0_argfix_gs100"; RM_TP=1 ;;
    qwen3-4B-rl-neutral-j35B-fakepen0)    RM_PATH="$MERGED/sender_qwen3-4B_neutral_j35B_fakepen0_gs100";   RM_TP=1 ;;
    qwen3-8B-rl-neutral-j35B-fakepen0)    RM_PATH="$MERGED/sender_qwen3-8B_neutral_j35B_fakepen0_gs100";   RM_TP=1 ;;
    # fmt0p1 neutral-juror arm (prior 0.5), counterpart of the stubborn fmt0p1 keys. Evaluate with
    # PROFILES=neutral and PROMPT=base so the eval juror matches the training one.
    qwen3-4B-rl-neutral-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-4B-rl-neutral-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-8B-rl-neutral-j35B-fakepen0-fmt0p1-gs50)  RM_PATH="$MERGED/sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1_gs50";  RM_TP=1 ;;
    qwen3-8B-rl-neutral-j35B-fakepen0-fmt0p1-gs100) RM_PATH="$MERGED/sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1_gs100"; RM_TP=1 ;;
    qwen3-4B-instruct-base)               RM_PATH="$TC/Qwen3-4B-Instruct-2507";                            RM_TP=1 ;;
    qwen3-8B-base)                        RM_PATH="$MODELS_DIR/Qwen3-8B";                                  RM_TP=1 ;;
    qwen3-14B-base)                       RM_PATH="$TC/Qwen3-14B";                                         RM_TP=1 ;;
    qwen3.5-35B)                          RM_PATH="$TC/Qwen3.5-35B-A3B";                                   RM_TP=2 ;;
    *)                                    RM_PATH="";                                                      RM_TP=1 ;;
  esac
}

# *_MODEL_PATH bypasses resolve_model, but a key matching a pattern branch still runs its validation
# (the ill11 branch exits on an unknown arm). _ALLOW_UNKNOWN makes an unresolvable key non-fatal when
# an override is present. The `if`s reset it to 0 before the receiver call, so the sender's 1 cannot
# hide a real receiver typo. (Not a `set -e` issue: bash exempts non-final commands of an `&&` list.)
if [ -n "${SENDER_MODEL_PATH:-}" ]; then _ALLOW_UNKNOWN=1; else _ALLOW_UNKNOWN=0; fi
resolve_model "$SENDER";   SPATH="${SENDER_MODEL_PATH:-$RM_PATH}";   STP="${SENDER_TP:-$RM_TP}"
if [ -n "${RECEIVER_MODEL_PATH:-}" ]; then _ALLOW_UNKNOWN=1; else _ALLOW_UNKNOWN=0; fi
resolve_model "$RECEIVER"; RPATH="${RECEIVER_MODEL_PATH:-$RM_PATH}"; RTP="${RECEIVER_TP:-$RM_TP}"
_ALLOW_UNKNOWN=0
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

# --- endpoint routing: the game stage gets the sender endpoint as CLI args; the receiver and the
#     fabrication-audit judge both use rl.receiver_client (RECEIVER_HOST/PORT/MODEL_ID, the training
#     transport and sampling), so no SGLANG_* routing is needed. ---
export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$RPORT" RECEIVER_MODEL_ID="$RECEIVER"
export PYTHON="$INFER_PYTHON"

FULL_TEMPLATE="datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json"
SENDER_SLUG="${SENDER//\//-}"; RECV_SLUG="${RECEIVER//\//-}"
# Mixture-of-receivers routing for rl.receiver_client (unset = all games on the local receiver).
# Two-way: RECEIVER_MIX_FRAC=0 (all local) or 1 (all API), with RECEIVER_MIX_MODEL set (routing is
# gated on it). N-way (RECEIVER_MIX_SPEC): one submission per arm with RECEIVER_MIX_PIN=<label>;
# eval rows are never stamped by 10H, so the pin decides every game. The judge/fabrication audit uses
# chat(), always local. An API arm calls RECEIVER_MIX_GATEWAY_URL, which defaults to GATEWAY_URL.
export RECEIVER_MIX_SPEC="${RECEIVER_MIX_SPEC:-}"
export RECEIVER_MIX_PIN="${RECEIVER_MIX_PIN:-}"
export RECEIVER_MIX_MODEL="${RECEIVER_MIX_MODEL:-}"
export RECEIVER_MIX_FRAC="${RECEIVER_MIX_FRAC:-0.5}"
export RECEIVER_MIX_SEED="${RECEIVER_MIX_SEED:-2026}"
# Only an explicitly set RECEIVER_MIX_GATEWAY_URL (e.g. a relay) has its host added to NO_PROXY
# below; the GATEWAY_URL default keeps its usual proxy route.
_MIX_URL_SET="${RECEIVER_MIX_GATEWAY_URL:-}"
export RECEIVER_MIX_GATEWAY_URL="${RECEIVER_MIX_GATEWAY_URL:-${GATEWAY_URL:-}}"
export RECEIVER_MIX_KEY_ENV="${RECEIVER_MIX_KEY_ENV:-GATEWAY_API_KEY}"
if [ -n "$RECEIVER_MIX_SPEC" ]; then
  # Validate the pin with the runtime's own parser before any GPU work: a typo'd pin would silently
  # fall back to the local juror under a hosted juror's file name (e.g. __recv_grok-4-1-fast-reasoning).
  # Prints "<pin slug>|<1 if a hosted arm will be played, else 0>". That flag, not a non-empty
  # gateway URL, gates the API key and URL checks: a hosted arm with no key returns empty completions
  # instead of failing, giving a file full of parse failures.
  _MIX_INFO="$("$INFER_PYTHON" -c 'import os,sys,rl.receiver_mix as m
spec = os.environ["RECEIVER_MIX_SPEC"]; p = os.environ.get("RECEIVER_MIX_PIN", "").strip()
L = m.labels_of(spec)
if p and p not in L: sys.exit(f"RECEIVER_MIX_PIN={p!r} is not one of {L}")
hosted = ([p] if p != m.LOCAL else []) if p else m.hosted_labels(spec)
print((m.slug(p) if p else "") + "|" + ("1" if hosted else "0"))')" \
    || { echo "ERROR: bad RECEIVER_MIX_SPEC/RECEIVER_MIX_PIN: $_MIX_INFO"; exit 1; }
  _MIX_PIN_SLUG="${_MIX_INFO%%|*}"; _MIX_HOSTED="${_MIX_INFO##*|}"
  if [ "$RECEIVER_MIX_GATEWAY_URL" = local ]; then
    export RECEIVER_MIX_GATEWAY_URL="http://127.0.0.1:${RPORT}/v1/chat/completions"
    export NO_PROXY="${NO_PROXY:-},127.0.0.1"; export no_proxy="$NO_PROXY"
  elif [ "$_MIX_HOSTED" = 1 ]; then
    _MIX_KEY="$(eval echo "\${${RECEIVER_MIX_KEY_ENV}:-}")"
    if [ -z "$_MIX_KEY" ]; then
      eval "$(grep -m1 "^export ${RECEIVER_MIX_KEY_ENV}=" ~/.bashrc 2>/dev/null || true)"
      _MIX_KEY="$(eval echo "\${${RECEIVER_MIX_KEY_ENV}:-}")"
    fi
    [ -n "$_MIX_KEY" ] || { echo "ERROR: RECEIVER_MIX_SPEC selects a hosted arm but \$${RECEIVER_MIX_KEY_ENV} is empty"; exit 1; }
    export "${RECEIVER_MIX_KEY_ENV}"
    [ -n "$RECEIVER_MIX_GATEWAY_URL" ] || { echo "ERROR: RECEIVER_MIX_SPEC selects a hosted arm but GATEWAY_URL is unset (export the gateway's chat-completions URL, or set RECEIVER_MIX_GATEWAY_URL)"; exit 1; }
    if [ -n "$_MIX_URL_SET" ]; then
      _MIX_HOST="$(echo "$RECEIVER_MIX_GATEWAY_URL" | sed -E 's#^[a-zA-Z]+://([^/:]+).*#\1#')"
      export NO_PROXY="${NO_PROXY:-},$_MIX_HOST"; export no_proxy="$NO_PROXY"
    fi
  fi
  # The pinned arm must reach the result filename, or two arms' evals land on one file.
  [ -n "$_MIX_PIN_SLUG" ] && RECV_SLUG="${RECV_SLUG}__recv_${_MIX_PIN_SLUG}"
  echo "[oldbailey-rl] receiver mixture spec='$RECEIVER_MIX_SPEC' pin='${RECEIVER_MIX_PIN:-<hash-split>}' seed=$RECEIVER_MIX_SEED"
fi
if [ -n "$RECEIVER_MIX_MODEL" ]; then
  # Sentinel 'local' => the co-served receiver endpoint (loopback; no key needed).
  if [ "$RECEIVER_MIX_GATEWAY_URL" = local ]; then
    export RECEIVER_MIX_GATEWAY_URL="http://127.0.0.1:${RPORT}/v1/chat/completions"
    export NO_PROXY="${NO_PROXY:-},127.0.0.1"; export no_proxy="$NO_PROXY"
  else
    _MIX_KEY="$(eval echo "\${${RECEIVER_MIX_KEY_ENV}:-}")"
    if [ -z "$_MIX_KEY" ]; then
      eval "$(grep -m1 "^export ${RECEIVER_MIX_KEY_ENV}=" ~/.bashrc 2>/dev/null || true)"
      _MIX_KEY="$(eval echo "\${${RECEIVER_MIX_KEY_ENV}:-}")"
    fi
    [ -n "$_MIX_KEY" ] || { echo "ERROR: RECEIVER_MIX_MODEL set but \$${RECEIVER_MIX_KEY_ENV} is empty"; exit 1; }
    export "${RECEIVER_MIX_KEY_ENV}"
    [ -n "$RECEIVER_MIX_GATEWAY_URL" ] || { echo "ERROR: RECEIVER_MIX_MODEL set but GATEWAY_URL is unset (export the gateway's chat-completions URL, or set RECEIVER_MIX_GATEWAY_URL)"; exit 1; }
    if [ -n "$_MIX_URL_SET" ]; then
      _MIX_HOST="$(echo "$RECEIVER_MIX_GATEWAY_URL" | sed -E 's#^[a-zA-Z]+://([^/:]+).*#\1#')"
      export NO_PROXY="${NO_PROXY:-},$_MIX_HOST"; export no_proxy="$NO_PROXY"
    fi
  fi
  _MIX_SLUG="$(echo "$RECEIVER_MIX_MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
  # The mixture arm must reach the result filename, or an all-API eval lands on the local one's file.
  RECV_SLUG="${RECV_SLUG}__mix_${_MIX_SLUG}${RECEIVER_MIX_FRAC//./p}"
  echo "[oldbailey-rl] receiver mixture arm: $RECEIVER_MIX_MODEL frac=$RECEIVER_MIX_FRAC seed=$RECEIVER_MIX_SEED"
fi
RES_DIR="experiments/results/old-bailey/${SENDER_SLUG}"

profile_prior() {  # profile -> the juror's stated prior P(guilty) (the persuasion baseline)
  case "$1" in
    stubborn)  echo 0.1 ;;
    *)         echo 0.5 ;;
  esac
}

VAL_ARG=()
if [ "$SPLIT" != "all" ] && [ -n "$VAL_PARQUET" ]; then
  if [ -f "$VAL_PARQUET" ]; then
    VAL_ARG=(--val-parquet "$VAL_PARQUET")
    echo "[oldbailey-rl] binding $SPLIT split to parquet: $VAL_PARQUET"
  else
    echo "[oldbailey-rl] WARN: VAL_PARQUET not found ($VAL_PARQUET); falling back to seed-42 shuffle"
  fi
fi

IDS_ARG=()
if [ -n "$IDS_FILE" ]; then
  [ -f "$IDS_FILE" ] || { echo "[oldbailey-rl] ERROR: IDS_FILE not found: $IDS_FILE"; exit 1; }
  IDS_ARG=(--ids-file "$IDS_FILE" --ids-key "$IDS_KEY")
  echo "[oldbailey-rl] game-id restriction: $IDS_FILE [$IDS_KEY]"
fi

echo "=== [oldbailey-rl] sender=$SENDER receiver=$RECEIVER split=$SPLIT profiles='$PROFILES' games=$END_IDX rounds=$NUM_STEPS ==="
for PROFILE in $PROFILES; do
  echo "------------------------------------------------------------------"
  echo "[run] profile=$PROFILE (RL-parity rollout, split=$SPLIT)"
  RES="$RES_DIR/${PROFILE}_oldbailey_rlrollout_${SPLIT}${RES_TAG}__recv_${RECV_SLUG}.json"
  "$INFER_PYTHON" evaluation/rl_rollout.py --receiver-api sglang --domain old-bailey --profile "$PROFILE" --split "$SPLIT" \
    "${VAL_ARG[@]}" --sender-prompt "$PROMPT" "${IDS_ARG[@]}" \
    --sender-host 127.0.0.1 --sender-port "$SPORT" --sender-name "$SENDER" \
    --end-idx "$END_IDX" --num-rounds "$NUM_STEPS" --max-workers "$MAX_WORKERS" \
    --out "$RES"
  [ -f "$RES" ] || { echo "[run] ERROR: result not found: $RES"; ls -la "$RES_DIR" || true; exit 1; }

  PRIOR="$(profile_prior "$PROFILE")"
  echo "[eval] profile=$PROFILE persuasion metrics (prior=$PRIOR) -> ${RES}.metrics.json"
  "$INFER_PYTHON" evaluation/old_bailey/evaluate_oldbailey.py \
    --result-file "$RES" \
    --model "$SENDER" \
    --receiver-config "$PROFILE" \
    --prior "$PRIOR" \
    --rounds "$NUM_STEPS" \
    --no-wandb

  if [ "$AUDIT_FAB" = 1 ]; then
    echo "[eval] profile=$PROFILE fabrication audit (35B judge, training instrument) -> ${RES%.json}.fabrication.json"
    "$INFER_PYTHON" evaluation/audit_fabrications.py --domain old-bailey \
      --result-file "$RES" \
      --full-template "$FULL_TEMPLATE" \
      --model "$SENDER" \
      --receiver-config "$PROFILE" \
      --max-workers "$MAX_WORKERS" \
      --no-wandb
  else
    echo "[eval] profile=$PROFILE fabrication audit SKIPPED (AUDIT_FAB=0; collection-only run)"
  fi

  # Optional 42-way strategy audit (STRATEGY_AUDIT=1) with the co-served 35B judge. Game-level (no
  # --per-turn): one 42-vector per game, which is what the analyses read. -> <result>.strategy_audit.json
  if [ "${STRATEGY_AUDIT:-0}" = 1 ]; then
    echo "[eval] profile=$PROFILE strategy audit (42-way, 35B judge) -> ${RES%.json}.strategy_audit.json"
    "$INFER_PYTHON" -m rl.strategy_audit.run --oldbailey-result "$RES" \
      --max-workers "$MAX_WORKERS"
  fi
done

_STEM="$RES_DIR/<profile>_oldbailey_rlrollout_${SPLIT}${RES_TAG}__recv_${RECV_SLUG}"
echo "[done] Old Bailey in-domain RL eval complete for sender=$SENDER (split=$SPLIT)."
echo "       metrics:     ${_STEM}.json.metrics.json   (the evaluator appends to the result file name)"
echo "       fabrication: ${_STEM}.fabrication.json"
