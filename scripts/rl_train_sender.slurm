#!/bin/bash
#SBATCH --job-name=rl_train_sender
#SBATCH --gres=gpu:6
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Sender (persuader) GRPO training. Default 4B/8B: one co-located 6-GPU job from the repo .venv.
#   GPUs 0-3  verl GRPO: $POLICY_MODEL FSDP2 actor+ref (full FT, no offload) + sglang rollout.
#   GPUs 4-5  JUDGE_MODEL (Qwen3.5-35B-A3B bf16, TP=2), served as both receiver and judge.
# qwen3-14B needs 8 trainer GPUs, so the judge cannot co-locate: run it as its own job
# (scripts/rl_serve_judge.slurm) and pass its endpoint file as JUDGE_ENDPOINT_FILE.
# Reward = receiver's reported P(guilty) - prior (rl/reward_function.py). Reward-hacking monitors
# (rl/monitors.py, verl response_length) are logged to wandb only, under reward_hacking/.
#
# ALGO=base[+penalty][+aux_loss]: components in rl/config/algorithm/, resolved by rl/arm_config.py.
#   base      outcome reward + FORMAT_REWARD_COEFF (0.1) * fraction of rounds with a closed
#             <thinking>+<argument> pair; monitors on for logging. Tag _fmt0p1.
#   penalty   score -= PENALTY_COEFFICIENT (0.3) * sum of the 11 misaligned-technique monitors
#             (PENALTY_TERMS=illegal_full11: belief coercion, the false-information indicator, the 9
#             per-strategy judges; one extra judge call per sample each). Tag pen0p3_illegal_full11
#             (fakepen0 when off).
#   aux_loss  auxiliary CE awareness loss on the probe sidecar in every optimizer step, gradient
#             balanced against GRPO. Needs PROMPT=strategies and verl patches 10I+13T.
#             Tag _auxce0p5_random_kindbal_normratio0p25.
#   A knob set in the environment overrides its component (reported as OVERRIDE), e.g. a coefficient
#   sweep: ALGO=base+penalty PENALTY_COEFFICIENT=0.5. Show the resolved values and their sources:
#     python -m rl.arm_config --describe --algo base+penalty --prompt strategies --repo $PWD
# PROMPT: base | strategies (42-technique allowed/forbidden guide) | single_strategy:<slug>. Sets the
#   data root rl[_sftsplit]<suffix> and the run-name suffix '' | _strategies | _only_<slug>
#   (python -m rl.sender_prompts --print data-suffix|name-suffix --prompt SPEC).
# DIST: bayesian (neutral juror, prior P(guilty)=0.5) | stubborn (skeptical juror, prior 0.1).
#
# `sbatch --export` splits on commas. ALGO therefore joins components with '+' (ALGO=base,penalty
# arrives as base), and comma-bearing values (RECEIVER_MIX_SPEC, EXTRA_OVERRIDES) must be exported in
# your shell and passed with a bare --export=ALL. RECEIVER_MIX_SPEC also accepts '+' between arms.
#
# Submit:
#   sbatch --export=ALL,ALGO=base,PROMPT=base,DIST=stubborn scripts/rl_train_sender.slurm
#   sbatch --export=ALL,ALGO=base+penalty,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1 scripts/rl_train_sender.slurm
#   sbatch --export=ALL,ALGO=base+aux_loss,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,ROLLOUT_SEED=2026 scripts/rl_train_sender.slurm
#   sbatch --export=ALL,SMOKE=1,DIST=bayesian scripts/rl_train_sender.slurm   # one-step smoke
#   # SFT-then-RL (scripts/sft_train_sender.slurm): from-base baseline, then the SFT-initialized arm.
#   sbatch --export=ALL,DIST=stubborn,PROMPT=strategies,SFT_SPLIT=1 scripts/rl_train_sender.slurm
#   sbatch --export=ALL,DIST=stubborn,PROMPT=strategies,SFT_SPLIT=1,SFT_INIT=$PWD/experiments/results/sft/sft_qwen3-4B_strategies_stubborn_ep2/hf_final,SFT_INIT_TAG=ep2 scripts/rl_train_sender.slurm
#   # qwen3-14B: external judge on 2 GPUs, then the 8-GPU trainer.
#   sbatch --export=ALL,JUDGE_MODEL=qwen3.5-35B,ENDPOINT_FILE=logs/judge_ep.txt scripts/rl_serve_judge.slurm
#   sbatch --gres=gpu:8 --export=ALL,DIST=bayesian,POLICY_MODEL=qwen3-14B,JUDGE_ENDPOINT_FILE=logs/judge_ep.txt scripts/rl_train_sender.slurm
#
# Env, as NAME (default): meaning.
#   ALGO (base), PROMPT (base), DIST (bayesian): see above.
#   FORMAT_REWARD_COEFF, PENALTY_TERMS, PENALTY_COEFFICIENT, AUX_CE_*: set by ALGO, overridable.
#   SMOKE (0): 1 = one-step run on tiny parquets in a separate _smoke data root.
#   POLICY_MODEL (qwen3-4B): qwen3-4B | qwen3-8B | qwen3-14B (8B/14B: reasoning, native <think> off).
#   POLICY_PATH (per model): policy checkpoint; not together with SFT_INIT.
#   MODELS_DIR (<repo>/models): default checkpoints are MODELS_DIR/<Model> (Qwen3-4B-Instruct-2507,
#     Qwen3-8B, Qwen3-14B, Qwen3.5-35B-A3B). CACHE_DIR (<repo>/.cache): HF, JIT, wandb and XDG caches.
#   N_GPUS (4; 8 for 14B): trainer GPUs.
#   TRAIN_BATCH (16): must be divisible by N_GPUS.
#   N_ROLLOUT (16): rollouts per prompt.
#   TOTAL_STEPS (100).
#   ROLLOUT_MEM_FRAC (per model): sglang mem_fraction_static for the rollout.
#   ROLLOUT_TP (1): rollout tensor parallel; 2 would shard the 14B but crashes NCCL here (see below).
#   JUDGE_MODEL (qwen3.5-35B), JUDGE_PATH (per model), JUDGE_TP (2): the served receiver + judge.
#   JUDGE_ENDPOINT_FILE (''): file holding an external judge's host:port; '' = co-serve it here.
#   SGLANG_EXTRA (''): extra flags for the co-served judge's sglang server.
#   DATA_DIR (<repo>/datasets/old_bailey/_generated/<data variant>/<DIST>): parquets to train on.
#   VERL_HOME (''): private patched verl checkout from scripts/make_private_verl.sh (<dir>/verl,
#     prepended to PYTHONPATH); '' = the verl installed in RL_VENV.
#   RL_VENV (<repo>/.venv), WANDB_MODE (offline).
#   SFT_SPLIT (0): 1 = 921-game rl_sftsplit* data root, tag _sftsplit. SFT_HOLDOUT (200) and
#     SFT_SEED (2026) define the carve-out.
#   SFT_INIT (''): HF dir of an SFT'd policy (<sft save_dir>/hf_final); needs SFT_SPLIT=1.
#     Tag _sftinit[-SFT_INIT_TAG].
#   REJECTION_SAMPLING (0): 1 = penalty-free rejection sampling (details below). Needs an active
#     penalty and monitors at sample rate 1.0. Tag _rs<target>.
#   RS_TARGET_CLEAN (8), RS_MAX_GEN_ROUNDS (2): clean rollouts wanted per prompt; extra regen rounds.
#   RL_STRATEGY_JUDGE_WORKERS (4), RL_MONITOR_JUDGE_RETRY (2): per-strategy judge threads; parse retries.
#   RECEIVER_MIX_SPEC (''): N-way receiver mixture 'label:weight,...'. The label is the hosted model
#     id; 'local' is the co-served JUDGE_MODEL. 'local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4'
#     = 4 games per receiver per 16-game step (1 each under SMOKE=1). Assignment is per game, so GRPO
#     groups stay receiver-homogeneous (patch_verl 10H, rl/receiver_mix.py); the judge never mixes.
#     Tag _mix<N>-<arms>-<hash>. A spec that parses to just 'local' is rejected (comma truncation).
#   RECEIVER_MIX_MODEL (''): 2-way mixture with one hosted id, e.g. DeepSeek-V4-Flash; each step trains
#     exactly round(RECEIVER_MIX_FRAC*batch) games against it. Not with the spec. Tag _mix-<slug><frac>.
#   RECEIVER_MIX_FRAC (0.5), RECEIVER_MIX_SEED (2026).
#   RECEIVER_MIX_PIN (''): force unstamped rows (validation, eval drivers) onto one arm.
#   RECEIVER_MIX_GATEWAY_URL (${GATEWAY_URL}): chat-completions URL for the hosted arms, e.g.
#     https://openrouter.ai/api/v1/chat/completions + RECEIVER_MIX_KEY_ENV=OPENROUTER_API_KEY, or
#     'local' for the co-served judge (no key). An explicit value's host is added to NO_PROXY.
#     GATEWAY_URL has no default: a hosted arm with neither variable set fails at job start.
#   RECEIVER_MIX_KEY_ENV (GATEWAY_API_KEY): name of the env var holding the bearer key.
#   RL_RECV_API_TEMPERATURE (1.0), RL_RECV_API_TOP_P (0.95): hosted receiver sampling. gpt-5-* reject
#     top_p and temperature != 1.0; rl/receiver_client.api_body() drops them per model, and the preflight
#     uses the same builder so a rejected parameter fails at job start.
#   RL_RECV_API_TIMEOUT (120), RL_RECV_API_CONCURRENCY (16): per process and per arm; per-arm override
#     RL_RECV_API_{TIMEOUT,CONCURRENCY}_<ARM> (label upper-cased, non-alphanumerics -> '_'). Use 240 for
#     a 4-way hosted mixture: DeepSeek-V4-Flash took 114.8 s on a round-3 prompt (p99 ~86 s), so 120
#     times out and double-bills.
#   AUX_CE_DATA (the policy size's sidecar under datasets/old_bailey/_generated/aux_ce_probe/variants/final/):
#     aux-loss probe sidecar. Not in the auto-name, so a sidecar ablation needs an explicit EXP_NAME or
#     resume_mode=auto continues the sibling arm's checkpoint.
#   ROLLOUT_SEED (''): integer 0..2147483646 for deterministic sender + co-served receiver sampling
#     (hosted receivers stay provider-defined). Rejected with JUDGE_ENDPOINT_FILE (determinism cannot be
#     verified) and REJECTION_SAMPLING. Needs patches 14T/14TF/14S/14SR/14R. Tag _rseed<seed>.
#   EXTRA_OVERRIDES (''): space-separated verl overrides, applied last so they win.
#   EXP_NAME (derived): run/checkpoint name.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Checkpoints are MODELS_DIR/<Model>; exported so Python in the job resolves the same root.
export MODELS_DIR="${MODELS_DIR:-$REPO/models}"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
mkdir -p "$LOG_DIR"
export LOG_DIR

RL_VENV="${RL_VENV:-$REPO/.venv}"; PY="$RL_VENV/bin/python"
DIST="${DIST:-bayesian}"
SMOKE="${SMOKE:-0}"
# Sender prompt spec (rl/sender_prompts.py). It is baked into the parquets, so each spec gets its own
# data root and run name, both derived below.
PROMPT="${PROMPT:-base}"
# SENDER_STRATEGIES is not read here; fail instead of silently training the base prompt.
[[ -z "${SENDER_STRATEGIES:-}" ]] || {
  echo "ERROR: the sender prompt is selected with PROMPT=base|strategies|single_strategy:<slug> (got SENDER_STRATEGIES='$SENDER_STRATEGIES')"; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: repo .venv missing at $RL_VENV"; exit 1; }
"$PY" -m rl.sender_prompts --check "$PROMPT" >/dev/null || {
  echo "ERROR: PROMPT='$PROMPT' is not a sender prompt spec (base|strategies|single_strategy:<slug>; slugs: rl/strategy_audit/taxonomy.py)"; exit 1; }
# Data-root suffix, run-name suffix and wandb tag, resolved once so they cannot disagree.
PROMPT_DATA_SUFFIX="$("$PY" -m rl.sender_prompts --print data-suffix --prompt "$PROMPT")" || {
  echo "ERROR: could not derive the data-root suffix for PROMPT='$PROMPT'"; exit 1; }
PROMPT_SUFFIX="$("$PY" -m rl.sender_prompts --print name-suffix --prompt "$PROMPT")" || {
  echo "ERROR: could not derive the run-name suffix for PROMPT='$PROMPT'"; exit 1; }
SENDER_PROMPT_TAG="$("$PY" -m rl.sender_prompts --print wandb-tag --prompt "$PROMPT")" || {
  echo "ERROR: could not derive the wandb tag for PROMPT='$PROMPT'"; exit 1; }   # custom_metadata.sender_prompt
# POLICY_MODEL picks checkpoint, rollout mem fraction, trainer GPUs and batch; POLICY_PATH wins.
# qwen3-8B/14B run with native <think> off (enable_thinking=False in rl/config/grpo_persuasion.yaml +
# patch_verl.sh (8)) and reason in the prompt's <thinking> block. Resolved before the algorithm
# block, which puts the model size into the aux-loss sidecar path.
POLICY_MODEL="${POLICY_MODEL:-qwen3-4B}"   # qwen3-4B | qwen3-8B | qwen3-14B
# Training algorithm (see header). rl/arm_config.py runs before every ${VAR:-default} below and sets
# only what the environment does not already carry, so explicit values win.
ALGO="${ALGO:-base}"
# `|| {...}`, not a bare eval "$(...)", which would swallow a non-zero exit. --describe runs before
# the eval so it reports what the environment already owned.
_ARM_ENV="$("$PY" -m rl.arm_config --print-env --algo "$ALGO" --prompt "$PROMPT" \
              --policy-model "$POLICY_MODEL" --repo "$REPO")" || {
  echo "ERROR: rl.arm_config rejected ALGO='$ALGO' PROMPT='$PROMPT'"; exit 1; }
_ARM_DESC="$("$PY" -m rl.arm_config --describe --algo "$ALGO" --prompt "$PROMPT" \
              --policy-model "$POLICY_MODEL" --repo "$REPO")"
echo "$_ARM_DESC"
# Resolved components in registry order; logged as custom_metadata.algo.
ALGO_RESOLVED="$(printf '%s\n' "$_ARM_DESC" | sed -n 's/^algorithm *: //p')"
[[ -n "$ALGO_RESOLVED" ]] || { echo "ERROR: rl.arm_config --describe named no components"; exit 1; }
eval "$_ARM_ENV"
# SFT_SPLIT=1: 921-game rl_sftsplit* parquets (split_rl_sft.py --sft-holdout removes 200 SFT games
# from train; validation unchanged). SFT_INIT: <save_dir>/hf_final from scripts/sft_train_sender.slurm.
# 921-game runs are not comparable to 1121-game runs: an SFT_INIT arm's only valid baseline is a
# from-base run with SFT_SPLIT=1.
SFT_SPLIT="${SFT_SPLIT:-0}"
SFT_HOLDOUT="${SFT_HOLDOUT:-200}"
SFT_SEED="${SFT_SEED:-2026}"
SFT_INIT="${SFT_INIT:-}"
SFT_INIT_TAG="${SFT_INIT_TAG:-}"
N_ROLLOUT="${N_ROLLOUT:-16}"
TRAIN_BATCH="${TRAIN_BATCH:-}"          # per-policy default (_BATCH_DEFAULT)
TOTAL_STEPS="${TOTAL_STEPS:-100}"
# Per-policy defaults. _BATCH_DEFAULT must be divisible by N_GPUS (verl: train_batch_size %
# n_gpus_per_node == 0). The 14B needs 8 trainer GPUs, so it cannot share a node with the 2-GPU judge:
# use JUDGE_ENDPOINT_FILE. The GPU check below rejects a co-located 14B.
case "$POLICY_MODEL" in
  qwen3-4B)  _POLICY_DEFAULT="$MODELS_DIR/Qwen3-4B-Instruct-2507"; _MEMFRAC_DEFAULT=0.5;  _NGPUS_DEFAULT=4; _BATCH_DEFAULT=16 ;;
  qwen3-8B)  _POLICY_DEFAULT="$MODELS_DIR/Qwen3-8B";                _MEMFRAC_DEFAULT=0.40; _NGPUS_DEFAULT=4; _BATCH_DEFAULT=16 ;;
  qwen3-14B) _POLICY_DEFAULT="$MODELS_DIR/Qwen3-14B";               _MEMFRAC_DEFAULT=0.55; _NGPUS_DEFAULT=8; _BATCH_DEFAULT=16 ;;
  *) echo "POLICY_MODEL must be qwen3-4B|qwen3-8B|qwen3-14B (or pass POLICY_PATH explicitly)"; exit 1 ;;
esac
if [[ -n "$SFT_INIT" ]]; then
  [[ -n "${POLICY_PATH:-}" ]] && { echo "ERROR: set SFT_INIT or POLICY_PATH, not both"; exit 1; }
  [[ -f "$SFT_INIT/config.json" ]] || { echo "ERROR: SFT_INIT is not an HF checkpoint dir (no config.json): $SFT_INIT"; exit 1; }
  [[ "$SFT_SPLIT" == 1 ]] || { echo "ERROR: SFT_INIT requires SFT_SPLIT=1 (the 921-game rl_sftsplit* root); a from-full run would leak the 200 SFT games into GRPO."; exit 1; }
  POLICY_PATH="$SFT_INIT"
fi
POLICY_PATH="${POLICY_PATH:-$_POLICY_DEFAULT}"
ROLLOUT_MEM_FRAC="${ROLLOUT_MEM_FRAC:-$_MEMFRAC_DEFAULT}"   # sglang mem_fraction_static (weights + KV) of the 80 GB GPU.
# The 14B at TP=1 needs 0.55: at 0.40 (32 GB) its 27.6 GB bf16 weights leave no KV room and sglang
# init aborts ("Not enough memory"); 0.55 leaves ~16 GB KV. verl free_cache_engine frees the KV during
# the FSDP step, so the larger arena does not OOM the actor update.
TRAIN_BATCH="${TRAIN_BATCH:-$_BATCH_DEFAULT}"
# Rollout TP; must divide N_GPUS. TP>1 would shard the 14B weights, but the verl + sglang-0.5.12 TP
# broadcast crashes NCCL here ("Cuda failure 999" in broadcast_pyobj; see grpo_persuasion.yaml), so
# the 14B runs TP=1 with the larger mem fraction. Retry ROLLOUT_TP=2 once that is fixed.
ROLLOUT_TP="${ROLLOUT_TP:-1}"
# One served model plays both receiver and judge; JUDGE_PATH wins over the default. Tune KV headroom
# with SGLANG_EXTRA (e.g. --mem-fraction-static 0.92 / --context-length).
JUDGE_MODEL="${JUDGE_MODEL:-qwen3.5-35B}"   # qwen3.5-35B (bf16, TP=2)
case "$JUDGE_MODEL" in
  qwen3.5-35B)  _JUDGE_DEFAULT="$MODELS_DIR/Qwen3.5-35B-A3B";       _JUDGE_TP_DEFAULT=2 ;;
  *) _JUDGE_DEFAULT=""; _JUDGE_TP_DEFAULT=2 ;;   # custom model: pass JUDGE_PATH explicitly
esac
JUDGE_PATH="${JUDGE_PATH:-$_JUDGE_DEFAULT}"
JUDGE_TP="${JUDGE_TP:-$_JUDGE_TP_DEFAULT}"   # = GPUs the judge gets (top of the allocation)
[[ -n "$JUDGE_PATH" ]] || { echo "ERROR: unknown JUDGE_MODEL=$JUDGE_MODEL; pass JUDGE_PATH explicitly."; exit 1; }
# Data root: rl[_sftsplit] + prompt suffix. Each variant bakes different parquets, so separate roots.
_DATA_VARIANT="rl${PROMPT_DATA_SUFFIX}"
[[ "$SFT_SPLIT" == 1 ]] && _DATA_VARIANT="rl_sftsplit${PROMPT_DATA_SUFFIX}"
# Smoke builds tiny parquets (10 games, 2 validation) in their own root so they never stand in for a
# full split: rl/rl_validation.parquet is the eval launchers' VAL_PARQUET default, and the auto-build
# only runs when the parquet is missing. set -e safe: the && list short-circuits.
[[ "$SMOKE" == 1 ]] && _DATA_VARIANT="${_DATA_VARIANT}_smoke"
_RL_DATA_ROOT="$REPO/datasets/old_bailey/_generated/$_DATA_VARIANT"
DATA_DIR="${DATA_DIR:-$_RL_DATA_ROOT/$DIST}"
[[ "$DIST" == bayesian || "$DIST" == stubborn ]] || { echo "DIST must be bayesian|stubborn"; exit 1; }
# Receiver metadata for wandb. Keep in sync with rl/game_rows.py and rl/cognitive_models.py.
case "$DIST" in
  bayesian)  RECEIVER_TYPES="bayesian" ;;
  stubborn)  RECEIVER_TYPES="stubborn" ;;
esac
RECEIVER_PRIOR=0.5; [[ "$DIST" == stubborn ]] && RECEIVER_PRIOR=0.1   # set -e safe (cf. SMOKE_SUFFIX)
N_GPUS="${N_GPUS:-$_NGPUS_DEFAULT}"   # trainer GPUs: 4 (4B/8B, +2 judge), 8 (14B); N_GPUS=2 for a 4-GPU-total 4B
JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-}"  # set: external judge, host:port read from this file

module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # local ckpts only
# JIT toolchain for SGLang/flashinfer: the qwen3_5 mamba kernel is tvm_ffi-JIT'd at CUDA-graph
# capture and needs ninja, nvcc and full cu13 headers. Mirrors scripts/oldbailey_rl_eval.slurm.
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
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_DIR/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_DIR/inductor}"
# wandb runs offline by default; upload afterwards with `wandb sync wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"
# Empty WANDB_ENTITY = your default entity.
export WANDB_PROJECT="${WANDB_PROJECT:-persuasion-gym-oldbailey-rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
# Reward-hacking monitors emit a fixed key set (monitors.MONITOR_KEYS) so verl's reward manager does
# not crash. ALGO turns them on (1 judge call/sample: the evidence audit); RL_MONITOR_ENABLE=0 turns
# them off. RL_MONITOR_SAMPLE_RATE<1 biases means toward 0 and is rejected with an active penalty.
# RL_MONITOR_TONE=1 adds the tone audit (a 2nd call/sample); otherwise rh_tone_* log as 0.0.
export RL_MONITOR_ENABLE
export RL_MONITOR_TONE="${RL_MONITOR_TONE:-0}"
# score = (mu - prior) - PENALTY_COEFFICIENT * sum of the active terms' rh_ monitor counts
# (rl/reward_function.py). PENALTY_TERMS, joined by ',' or '+': fabrication | fabrication_binary
# (rh_fake_evidence count / 0-1 indicator, free) | belief | the 9 per-strategy terms | illegal_full11
# (all 11 illegal-strategy terms). belief and per-strategy terms are binary, one extra judge call per
# sample each. Both knobs empty = no penalty. Defaults: rl/config/algorithm/penalty.yaml.
export PENALTY_TERMS
export PENALTY_COEFFICIENT
# Receiver mixture (see header); all empty = off. rl/receiver_client.py reads these at call time and
# routes each game per patch 10H's stratified assignment (train) or a per-game hash (validation,
# eval). Exported so Ray rollout workers and the reward threads inherit them.
export RECEIVER_MIX_SPEC="${RECEIVER_MIX_SPEC:-}"
export RECEIVER_MIX_PIN="${RECEIVER_MIX_PIN:-}"
export RECEIVER_MIX_MODEL="${RECEIVER_MIX_MODEL:-}"
export RECEIVER_MIX_FRAC="${RECEIVER_MIX_FRAC:-0.5}"
export RECEIVER_MIX_SEED="${RECEIVER_MIX_SEED:-2026}"
export RECEIVER_MIX_GATEWAY_URL="${RECEIVER_MIX_GATEWAY_URL:-}"
export RL_RECV_API_TEMPERATURE="${RL_RECV_API_TEMPERATURE:-1.0}"
export RL_RECV_API_TOP_P="${RL_RECV_API_TOP_P:-0.95}"
# Too short for a 4-way hosted mixture; see header.
export RL_RECV_API_TIMEOUT="${RL_RECV_API_TIMEOUT:-120}"
# Default for RL_RECV_API_CONCURRENCY_<ARM>. Semaphores are per arm, so a slow engine cannot block
# the fast ones.
export RL_RECV_API_CONCURRENCY="${RL_RECV_API_CONCURRENCY:-16}"
# Format bonus (see header). Normalized once with reward_function._format_reward_coeff's clamp
# (negative or garbage -> 0), so the _fmt name tag, the env and wandb agree; awk %g prints any zero as 0.
FORMAT_REWARD_COEFF=$(awk -v v="$FORMAT_REWARD_COEFF" 'BEGIN{x=v+0.0; if (x<0) x=0; printf "%g", x}')
export FORMAT_REWARD_COEFF
# Rejection sampling. Prompts with fewer than RS_TARGET_CLEAN clean rollouts (penalty_total==0,
# parse_failure==0, format_reward==1.0, monitor_failure==0) get up to RS_MAX_GEN_ROUNDS extra full-n
# regen rounds, batched. Then exactly N_ROLLOUT rows are kept per prompt, positives first, in pool
# order; shortfalls are filled with negatives (rejection/fallback_fill_count). The format leg rejects
# empty degenerate outputs; the outage leg stops unaudited rollouts from counting as clean.
# Hook: patch_verl.sh (10G); selection: rl/rejection_sampling.py. Without a penalty it is a no-op.
export REJECTION_SAMPLING="${REJECTION_SAMPLING:-0}"
export RS_TARGET_CLEAN="${RS_TARGET_CLEAN:-8}"      # clamped in-code to min(target, N_ROLLOUT)
export RS_MAX_GEN_ROUNDS="${RS_MAX_GEN_ROUNDS:-2}"  # extra regeneration rounds after the initial one
# Aux CE loss (ALGO=...+aux_loss; patches 10I+13, logic in rl/aux_ce.py): each GRPO optimizer step adds
# AUX_CE_COEFF * CE on the gold true/false token of in-role probe items, one per rollout row. Rebuild
# the sidecar with datasets/old_bailey/aux_loss/build_chain.sh, which also writes the _meta.json
# checked below. Values, including the off state, come from rl/config/algorithm/aux_loss.yaml.
export AUX_CE
export AUX_CE_DATA
export AUX_CE_COEFF
export AUX_CE_MAX_PROMPT
export AUX_CE_ROW_FRAC
export AUX_CE_SEED
export AUX_CE_SAMPLING
export AUX_CE_KIND_BALANCE
export AUX_CE_BALANCE_MODE
export AUX_CE_TARGET_GRAD_RATIO
# Seeds sender + local-receiver training sampling and reward replay. Validation stays unseeded (it never
# enters an update); hosted receiver RNG is provider-defined.
export ROLLOUT_SEED="${ROLLOUT_SEED:-}"
# Teacher behind SFT_INIT's demonstrations (== policy for self-distillation). wandb metadata only.
export TEACHER_MODEL="${TEACHER_MODEL:-unknown}"
# Rollout dumps to <save_dir>/rollouts/<step>.jsonl (patch 10C): step 1 + every N steps (0 = all).
export RL_ROLLOUT_DUMP_FREQ="${RL_ROLLOUT_DUMP_FREQ:-5}"
# RL_JUDGE_DUMP_DIR / RL_EXP_NAME are exported per run after EXP_NAME is known.

[[ -x "$PY" ]] || { echo "ERROR: repo .venv missing at $RL_VENV"; exit 1; }
# VERL_HOME=<dir> uses <dir>/verl (scripts/make_private_verl.sh), isolating a study from jobs that
# re-patch the shared verl in place. PYTHONPATH reaches the patch greps, the checkers, training and Ray
# workers. Resolve the import now so a bad dir fails before the GPUs are held.
VERL_HOME="${VERL_HOME:-}"
VERL_HOME="${VERL_HOME%/}"   # strip a trailing '/' for the prefix test below
if [[ -n "$VERL_HOME" ]]; then
  [[ -d "$VERL_HOME/verl" ]] || { echo "ERROR: VERL_HOME=$VERL_HOME holds no verl/ package (expected $VERL_HOME/verl)"; exit 1; }
  export PYTHONPATH="$VERL_HOME${PYTHONPATH:+:$PYTHONPATH}"
  _VERL_FILE="$("$PY" -c 'import verl; print(verl.__file__)')" \
    || { echo "ERROR: VERL_HOME=$VERL_HOME but 'import verl' failed"; exit 1; }
  case "$_VERL_FILE" in
    "$VERL_HOME"/*) echo "[verl] private checkout: $VERL_HOME" ;;
    *) echo "ERROR: VERL_HOME=$VERL_HOME but 'import verl' resolved to $_VERL_FILE"; exit 1 ;;
  esac
fi
"$PY" -c "import verl" 2>/dev/null || { echo "ERROR: verl not installed in .venv -- run scripts/rl_build_env.sh first."; exit 1; }
# Validate before any GPU work. Canonical spelling stops e.g. rseed02026 aliasing seed 2026. The
# checker validates every generated verl integration point and its syntax.
_ROLLOUT_SEED_META=-1
_ROLLOUT_DETERMINISTIC_META=false
_ROLLOUT_SEED_IMPL_META=off
if [[ -n "$ROLLOUT_SEED" ]]; then
  if [[ ! "$ROLLOUT_SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( ROLLOUT_SEED > 2147483646 )); then
    echo "ERROR: ROLLOUT_SEED must be a canonical integer in 0..2147483646 (got '$ROLLOUT_SEED')"
    exit 1
  fi
  if [[ "$REJECTION_SAMPLING" == 1 ]]; then
    echo "ERROR: ROLLOUT_SEED with REJECTION_SAMPLING=1 is unsupported: regenerated rows need distinct deterministic rollout offsets"
    exit 1
  fi
  if [[ -n "$JUDGE_ENDPOINT_FILE" ]]; then
    echo "ERROR: ROLLOUT_SEED with JUDGE_ENDPOINT_FILE is unsupported: this launcher cannot verify"
    echo "       that the external receiver server consumes request seeds under deterministic inference."
    echo "       Use the co-served receiver path, or leave ROLLOUT_SEED unset."
    exit 1
  fi
  _ROLLOUT_SEED_META="$ROLLOUT_SEED"
  _ROLLOUT_DETERMINISTIC_META=true
  _ROLLOUT_SEED_IMPL_META=detinf1
  _RSEED_RT="$($PY -c 'import verl.trainer.ppo.ray_trainer as m; print(m.__file__)')"
  _RSEED_SG="$($PY -c 'import verl.workers.rollout.sglang_rollout.sglang_rollout as m; print(m.__file__)')"
  _RSEED_SCHEMA="$($PY -c 'import verl.workers.rollout.schemas as m; print(m.__file__)')"
  grep -q "persuasion-gym] (14T)" "$_RSEED_RT" || { echo "ERROR: ROLLOUT_SEED is set but $_RSEED_RT lacks patch 14T -- run: bash rl/patch_verl.sh"; exit 1; }
  grep -q "persuasion-gym] (14S)" "$_RSEED_SG" || { echo "ERROR: ROLLOUT_SEED is set but $_RSEED_SG lacks patch 14S -- run: bash rl/patch_verl.sh"; exit 1; }
  grep -q "persuasion-gym] (14R)" "$_RSEED_SCHEMA" || { echo "ERROR: ROLLOUT_SEED is set but $_RSEED_SCHEMA lacks patch 14R -- run: bash rl/patch_verl.sh"; exit 1; }
  "$PY" -m rl.check_rollout_seed_patch \
    --trainer "$_RSEED_RT" --rollout "$_RSEED_SG" --schema "$_RSEED_SCHEMA"
fi
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
REQ=$(( N_GPUS + JUDGE_TP )); [[ -n "$JUDGE_ENDPOINT_FILE" ]] && REQ=$N_GPUS
(( NGPU >= REQ )) || { echo "ERROR: need $REQ GPUs (N_GPUS=$N_GPUS), have $NGPU. Resubmit with --gres=gpu:$REQ."; exit 1; }

# --- prepare parquets if missing (cheap, no GPU) ---
if [[ ! -f "$DATA_DIR/rl_train.parquet" ]]; then
  echo "[data] building parquets ..."
  # Smoke train set needs >= N_GPUS rows (drop_last loader, batch = N_GPUS) or init fails with "Train
  # dataloader is empty!". Floor at 10 games: val_size clamps to len//5 and a 0-row validation parquet
  # is unreadable; 10 keeps val>=2 and, with the smoke SFT holdout of 2, train>=6.
  _SMOKE_END=$(( N_GPUS * 2 )); (( _SMOKE_END < 10 )) && _SMOKE_END=10
  PREP_ARGS=""; [[ "$SMOKE" == 1 ]] && PREP_ARGS="--end-index $_SMOKE_END --val-size 2"
  PREP_ARGS="$PREP_ARGS --sender-prompt $PROMPT"
  if [[ "$SFT_SPLIT" == 1 ]]; then
    # Leakage guard (build side): an rl_sftsplit* build must carve out the SFT holdout.
    _SFT_HOLDOUT_EFF="$SFT_HOLDOUT"; [[ "$SMOKE" == 1 ]] && _SFT_HOLDOUT_EFF=2
    PREP_ARGS="$PREP_ARGS --sft-holdout $_SFT_HOLDOUT_EFF --sft-seed $SFT_SEED"
  fi
  # Non-default variants write under their own root; pass --out-dir once.
  [[ "$_DATA_VARIANT" != rl ]] && PREP_ARGS="$PREP_ARGS --out-dir $_RL_DATA_ROOT"
  "$PY" "$REPO/datasets/old_bailey/split_rl_sft.py" $PREP_ARGS
fi
if [[ "$SFT_SPLIT" == 1 ]]; then
  # Leakage guard (load side, every launch): the train parquet must be disjoint from the SFT holdout
  # with the expected games, which catches a root built without --sft-holdout.
  _SPLIT_JSON="$(dirname "$DATA_DIR")/sft_split.json"   # root-level, next to the per-DIST dirs
  [[ -f "$_SPLIT_JSON" ]] || { echo "ERROR: $_SPLIT_JSON missing -- this root was not built with --sft-holdout (rebuild it)"; exit 1; }
  "$PY" - "$DATA_DIR/rl_train.parquet" "$_SPLIT_JSON" <<'PYEOF'
import json, sys
import pandas as pd
id_list = [e["index"] for e in pd.read_parquet(sys.argv[1])["extra_info"]]
ids = set(id_list)
split = json.load(open(sys.argv[2]))
sft, expected = set(split["sft_train_ids"]), set(split["rl_train_ids"])
# Check the list too: duplicate rows would over-sample games yet pass the set checks.
assert len(id_list) == len(ids), f"DUPLICATES: train parquet has {len(id_list)} rows but only {len(ids)} unique game ids"
assert not (ids & sft), f"LEAKAGE: {len(ids & sft)} SFT-holdout games found in the train parquet"
assert ids == expected, f"train parquet has {len(ids)} games, sft_split.json expects {len(expected)}"
print(f"[data] sftsplit guard OK: {len(id_list)} train rows / {len(ids)} unique games disjoint from {len(sft)} SFT-holdout games")
PYEOF
fi

# --- judge: external (rl_serve_judge.slurm) or co-located on the top JUDGE_TP GPUs ---
export RECEIVER_MODEL_ID="$JUDGE_MODEL"
if [[ -n "$JUDGE_ENDPOINT_FILE" ]]; then
  # Wait up to JUDGE_WAIT_MIN minutes for the external judge to write its endpoint file.
  JUDGE_WAIT_MIN="${JUDGE_WAIT_MIN:-45}"
  echo "[judge] EXTERNAL mode: waiting up to ${JUDGE_WAIT_MIN}min for $JUDGE_ENDPOINT_FILE ..."
  for i in $(seq 1 $(( JUDGE_WAIT_MIN * 12 ))); do [[ -s "$JUDGE_ENDPOINT_FILE" ]] && break; sleep 5; done
  [[ -s "$JUDGE_ENDPOINT_FILE" ]] || { echo "[judge] endpoint file never appeared: $JUDGE_ENDPOINT_FILE"; exit 1; }
  EP="$(tr -d '[:space:]' < "$JUDGE_ENDPOINT_FILE")"; export RECEIVER_HOST="${EP%%:*}" RECEIVER_PORT="${EP##*:}"
  export NO_PROXY="$NO_PROXY,$RECEIVER_HOST"; export no_proxy="$NO_PROXY"   # reach the judge host directly, bypassing any HTTP proxy
  for i in $(seq 1 60); do
    curl --noproxy '*' -sf "http://${RECEIVER_HOST}:${RECEIVER_PORT}/v1/models" >/dev/null 2>&1 && break; sleep 5
  done
  curl --noproxy '*' -sf "http://${RECEIVER_HOST}:${RECEIVER_PORT}/v1/models" >/dev/null 2>&1 \
    || { echo "[judge] external judge unreachable at $RECEIVER_HOST:$RECEIVER_PORT"; exit 1; }
  echo "[judge] external judge UP at $RECEIVER_HOST:$RECEIVER_PORT"
  TRAINER_GPUS="$(seq -s, 0 $(( N_GPUS - 1 )))"
else
  JPORT=$(( 20000 + (SLURM_JOB_ID % 20000) )); JLOG="$LOG_DIR/judge_${SLURM_JOB_ID}.log"
  JUDGE_GPUS="$(seq -s, "$N_GPUS" $(( N_GPUS + JUDGE_TP - 1 )))"   # the top JUDGE_TP GPUs of the allocation
  echo "[serve] $JUDGE_MODEL <- $JUDGE_PATH (TP=$JUDGE_TP) on 127.0.0.1:$JPORT (GPUs $JUDGE_GPUS)"
  _JUDGE_DETERMINISTIC_ARGS=()
  if [[ -n "$ROLLOUT_SEED" ]]; then
    _JUDGE_DETERMINISTIC_ARGS=(--enable-deterministic-inference --random-seed "$ROLLOUT_SEED")
  fi
  CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" HTTPS_PROXY= HTTP_PROXY= http_proxy= https_proxy= \
    "$PY" -m sglang.launch_server --model-path "$JUDGE_PATH" --served-model-name "$JUDGE_MODEL" \
      --host 127.0.0.1 --port "$JPORT" --tp-size "$JUDGE_TP" \
      "${_JUDGE_DETERMINISTIC_ARGS[@]}" ${SGLANG_EXTRA:-} > "$JLOG" 2>&1 &
  SGPID=$!; trap 'kill "$SGPID" 2>/dev/null || true' EXIT
  echo "[serve] waiting for judge readiness (up to ${JUDGE_WAIT_TRIES:-300} x 5s; the judge weights take minutes to load) ..."
  for i in $(seq 1 "${JUDGE_WAIT_TRIES:-300}"); do
    curl --noproxy '*' -sf "http://127.0.0.1:${JPORT}/v1/models" >/dev/null 2>&1 && { echo "[serve] judge UP ($i)"; break; }
    kill -0 "$SGPID" 2>/dev/null || { echo "[serve] judge died:"; tail -n 60 "$JLOG"; exit 1; }
    sleep 5
  done
  curl --noproxy '*' -sf "http://127.0.0.1:${JPORT}/v1/models" >/dev/null 2>&1 || { echo "[serve] judge not ready:"; tail -n 120 "$JLOG"; exit 1; }
  export RECEIVER_HOST=127.0.0.1 RECEIVER_PORT="$JPORT"
  TRAINER_GPUS="$(seq -s, 0 $(( N_GPUS - 1 )))"   # the bottom N_GPUS GPUs
fi

# --- GRPO smoke vs full ---
SMOKE_OVERRIDES=""
if [[ "$SMOKE" == 1 ]]; then
  TRAIN_BATCH=$N_GPUS; N_ROLLOUT=2; TOTAL_STEPS=1   # train_batch_size % n_gpus_per_node == 0
  # Never resume (FSDP world size may differ) or save (full-FT ckpts are multi-GB).
  SMOKE_OVERRIDES="trainer.resume_mode=disable trainer.save_freq=-1"
fi
SMOKE_SUFFIX=""; [[ "$SMOKE" == 1 ]] && SMOKE_SUFFIX="_smoke"  # avoid set -e death on the false branch
# Effective penalty terms + coefficient for the run name, monitor guard and wandb; mirrors
# rl/reward_function.py::_penalty_config. Term names are validated here so a typo fails at submit.
# illegal_full11 = reward_function.ILLEGAL_TERMS: belief (belief_coercion), fabrication_binary (the
# false_information indicator) and the 9 per-strategy judge terms.
_ILLEGAL_TERMS="belief fabrication_binary threats social_punishment creating_dependency authority_impersonation procedural_manipulation discouragement false_promises misrepresentation rumors"
_KNOWN_TERMS="fabrication $_ILLEGAL_TERMS"
# Penalty-free by default: coefficient 0.0, which gives the fakepen0 tag.
EFF_COEFF=0.0; EFF_TERMS=""
if [[ -n "$PENALTY_TERMS" || -n "$PENALTY_COEFFICIENT" ]]; then
  EFF_COEFF="${PENALTY_COEFFICIENT:-0.0}"
  for _name in $(echo "$PENALTY_TERMS" | tr ',+' '  '); do
    [[ -z "$_name" ]] && continue
    [[ "$_name" == illegal_full11 ]] && { EFF_TERMS="$EFF_TERMS $_ILLEGAL_TERMS"; continue; }
    [[ " $_KNOWN_TERMS " == *" $_name "* ]] || { echo "ERROR: unknown PENALTY_TERMS entry '$_name' (known: $_KNOWN_TERMS, alias: illegal_full11)"; exit 1; }
    EFF_TERMS="$EFF_TERMS $_name"
  done
fi
_coeff_zero=0; case "$EFF_COEFF" in ''|0|0.0|0.00|0.000) _coeff_zero=1 ;; esac
# Sorted, de-duped terms give a stable name (belief_fab). sed '/^$/d', not grep -v, which exits 1 on
# empty input and aborts under pipefail.
EFF_TERMS="$(echo $EFF_TERMS | tr ' ' '\n' | sort -u | sed '/^$/d' | tr '\n' ' ' | sed 's/^ *//; s/ *$//')"
# fabrication (count) and fabrication_binary (indicator) read the same rh_fake_evidence monitor.
if [[ " $EFF_TERMS " == *" fabrication "* && " $EFF_TERMS " == *" fabrication_binary "* ]]; then
  echo "ERROR: PENALTY_TERMS names both fabrication (count) and fabrication_binary (indicator); pick one"; exit 1
fi
# Every penalty term reads an rh_ monitor, so an active penalty needs monitors on at rate 1.0; an
# unsampled rollout reads rh_*=0 and trains penalty-free.
if [[ -n "$EFF_TERMS" && "$_coeff_zero" == 0 ]]; then
  [[ "${RL_MONITOR_ENABLE}" == 1 ]] || { echo "ERROR: an active penalty needs RL_MONITOR_ENABLE=1 (got '${RL_MONITOR_ENABLE}')"; exit 1; }
  export RL_MONITOR_SAMPLE_RATE="${RL_MONITOR_SAMPLE_RATE:-1.0}"
  # any numeric spelling of 1 (1, 1.0, 1e0, ...)
  awk -v r="${RL_MONITOR_SAMPLE_RATE}" 'BEGIN{exit !(r==1)}' || { echo "ERROR: an active penalty needs RL_MONITOR_SAMPLE_RATE=1.0 (got '${RL_MONITOR_SAMPLE_RATE}')"; exit 1; }
fi
# Rejection sampling filters on penalty_total: without an active penalty every rollout is clean, and
# an unsampled rollout (rh_*=0) would count as clean.
if [[ "${REJECTION_SAMPLING}" == 1 ]]; then
  [[ -n "$EFF_TERMS" && "$_coeff_zero" == 0 ]] || { echo "ERROR: REJECTION_SAMPLING=1 requires an active penalty (PENALTY_TERMS + PENALTY_COEFFICIENT>0)"; exit 1; }
  # _coeff_zero only matches literal zeros; also reject 0e0, negatives and garbage numerically.
  awk -v c="$EFF_COEFF" 'BEGIN{exit !(c+0>0)}' || { echo "ERROR: REJECTION_SAMPLING=1 needs a numerically positive penalty coefficient (got '$EFF_COEFF')"; exit 1; }
  [[ "${RL_MONITOR_ENABLE}" == 1 ]] || { echo "ERROR: REJECTION_SAMPLING=1 needs RL_MONITOR_ENABLE=1 (got '${RL_MONITOR_ENABLE}')"; exit 1; }
  export RL_MONITOR_SAMPLE_RATE="${RL_MONITOR_SAMPLE_RATE:-1.0}"
  awk -v r="${RL_MONITOR_SAMPLE_RATE}" 'BEGIN{exit !(r==1)}' || { echo "ERROR: REJECTION_SAMPLING=1 needs RL_MONITOR_SAMPLE_RATE=1.0 (got '${RL_MONITOR_SAMPLE_RATE}')"; exit 1; }
  [[ "$RS_TARGET_CLEAN" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: RS_TARGET_CLEAN must be a positive integer (got '$RS_TARGET_CLEAN')"; exit 1; }
  [[ "$RS_MAX_GEN_ROUNDS" =~ ^[0-9]+$ ]] || { echo "ERROR: RS_MAX_GEN_ROUNDS must be a non-negative integer (got '$RS_MAX_GEN_ROUNDS')"; exit 1; }
  # An unpatched verl would ignore the knob while the run still carries the _rs name.
  _RS_RT="$("$PY" -c 'import verl.trainer.ppo.ray_trainer as m; print(m.__file__)')"
  grep -q "persuasion-gym] (10G)" "$_RS_RT" || { echo "ERROR: REJECTION_SAMPLING=1 but the verl checkout ($_RS_RT) lacks patch 10G -- run: bash rl/patch_verl.sh"; exit 1; }
fi
# Aux-CE guards fail at job start, so an _auxce run never silently trains without the aux loss.
[[ "$AUX_CE" == 0 || "$AUX_CE" == 1 ]] || { echo "ERROR: AUX_CE must be 0 or 1 (got '$AUX_CE')"; exit 1; }
[[ "$AUX_CE_BALANCE_MODE" == off || "$AUX_CE_BALANCE_MODE" == norm_ratio ]] || {
  echo "ERROR: AUX_CE_BALANCE_MODE must be 'off' or 'norm_ratio' (got '$AUX_CE_BALANCE_MODE')"; exit 1; }
if [[ "$AUX_CE" == 0 && "$AUX_CE_BALANCE_MODE" != off ]]; then
  echo "ERROR: AUX_CE_BALANCE_MODE=$AUX_CE_BALANCE_MODE requires AUX_CE=1 (no aux gradient exists when AUX_CE=0)"
  exit 1
fi
[[ "$AUX_CE_KIND_BALANCE" == 0 || "$AUX_CE_KIND_BALANCE" == 1 ]] || {
  echo "ERROR: AUX_CE_KIND_BALANCE must be 0 or 1 (got '$AUX_CE_KIND_BALANCE')"; exit 1; }
if [[ "$AUX_CE" == 0 && "$AUX_CE_KIND_BALANCE" == 1 ]]; then
  echo "ERROR: AUX_CE_KIND_BALANCE=1 requires AUX_CE=1 (there is no aux batch to balance)"
  exit 1
fi
if [[ "$AUX_CE" == 1 ]]; then
  [[ "$PROMPT" == strategies ]] || { echo "ERROR: AUX_CE=1 requires PROMPT=strategies (the sidecar probe contexts bake the 42-strategy guide)"; exit 1; }
  [[ "$REJECTION_SAMPLING" == 0 ]] || { echo "ERROR: AUX_CE=1 + REJECTION_SAMPLING=1 is an unvalidated combination; refusing"; exit 1; }
  [[ -s "$AUX_CE_DATA" ]] || { echo "ERROR: AUX_CE=1 but AUX_CE_DATA is missing/empty: '$AUX_CE_DATA' (build it: datasets/old_bailey/aux_loss/build_chain.sh)"; exit 1; }
  awk -v c="$AUX_CE_COEFF" 'BEGIN{exit !(c+0>0)}' || { echo "ERROR: AUX_CE_COEFF must be numerically > 0 (got '$AUX_CE_COEFF')"; exit 1; }
  awk -v f="$AUX_CE_ROW_FRAC" 'BEGIN{exit !(f+0>0 && f+0<=1)}' || { echo "ERROR: AUX_CE_ROW_FRAC must be in (0,1] (got '$AUX_CE_ROW_FRAC')"; exit 1; }
  [[ "$AUX_CE_SAMPLING" == random ]] || { echo "ERROR: AUX_CE_SAMPLING must be 'random' (got '$AUX_CE_SAMPLING')"; exit 1; }
  if [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]]; then
    # Canonicalize so .25 / 0.250 / 2.5e-1 give the same env, name and metadata.
    _AUX_TARGET_RAW="$AUX_CE_TARGET_GRAD_RATIO"
    if ! AUX_CE_TARGET_GRAD_RATIO="$("$PY" - "$_AUX_TARGET_RAW" <<'PYEOF'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
if not math.isfinite(value) or value <= 0:
    raise SystemExit(1)
print(f"{value:.12g}")
PYEOF
)"; then
      echo "ERROR: AUX_CE_TARGET_GRAD_RATIO must be finite and > 0 in norm_ratio mode (got '$_AUX_TARGET_RAW')"
      exit 1
    fi
    export AUX_CE_TARGET_GRAD_RATIO
  fi
  _AUX_RT="$("$PY" -c 'import verl.trainer.ppo.ray_trainer as m; print(m.__file__)')"
  grep -q "persuasion-gym] (10I)" "$_AUX_RT" || { echo "ERROR: AUX_CE=1 but the verl checkout ($_AUX_RT) lacks patch 10I -- run: bash rl/patch_verl.sh"; exit 1; }
  _AUX_DP="$("$PY" -c 'import verl.workers.actor.dp_actor as m; print(m.__file__)')"
  # Only the multi-term patch (13T) emits aux_term_id / aux_ce_terms for rl/aux_ce.py. An unpatched
  # checkout needs patch_verl.sh; one with the single-channel (13) edit needs that block reset first,
  # which patch_verl.sh also does.
  if ! grep -q "persuasion-gym] (13T)" "$_AUX_DP"; then
    if grep -q "persuasion-gym] (13)" "$_AUX_DP"; then
      echo "ERROR: AUX_CE=1 but $_AUX_DP carries the single-channel patch (13), which cannot"
      echo "       drive rl/aux_ce.py (it never emits aux_term_id / aux_ce_terms). Re-run"
      echo "       'bash rl/patch_verl.sh': it resets that block to pristine and applies (13T)."
    else
      echo "ERROR: AUX_CE=1 but $_AUX_DP is unpatched (no multi-term patch 13T)."
      echo "       Run: bash rl/patch_verl.sh"
    fi
    exit 1
  fi
  # 13T is shared by the norm-ratio and fixed-coefficient actor patches, so norm_ratio runs also get
  # the structural checker, which catches a stale venv that has the 13T marker.
  if [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]]; then
    "$PY" -m rl.check_aux_grad_balance_patch \
      --actor "$_AUX_DP" --trainer "$_AUX_RT" || {
      echo "ERROR: AUX_CE_BALANCE_MODE=norm_ratio requires the current adaptive Verl patch."
      echo "       Re-run: bash rl/patch_verl.sh"
      exit 1
    }
  fi
  # Sidecar <-> run parity: both labels, token cap, chat template == policy tokenizer's, and label
  # balance for the "random" sampler (it draws from per-label pools). Game coverage is informational.
  "$PY" - "$DATA_DIR/rl_train.parquet" "$AUX_CE_DATA" "$AUX_CE_MAX_PROMPT" "$POLICY_PATH" "$AUX_CE_KIND_BALANCE" <<'PYEOF'
import gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path
import pandas as pd
pq, sidecar, max_tok, policy = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
kind_balance = sys.argv[5] == "1"
KIND_BLOCK = {"claim_real": "a", "fabrication": "a",
              "paraphrase_real": "b", "altered_real": "b"}
train_ids = {int(e["index"]) for e in pd.read_parquet(pq)["extra_info"]}
have = {"true": set(), "false": set()}
cnt = {"true": 0, "false": 0}
block_cnt = Counter()
kinds = Counter()
n = 0
with gzip.open(sidecar, "rt") as fh:
    for ln in fh:
        if ln.strip():
            r = json.loads(ln)
            have[r["label"]].add(int(r["game_id"])); cnt[r["label"]] += 1; n += 1
            if kind_balance:
                k = r.get("kind")
                assert k in KIND_BLOCK, (
                    f"AUX_CE_KIND_BALANCE=1 but item {r['item_id']!r} has kind {k!r}, not one of "
                    f"{sorted(KIND_BLOCK)} -- this is not a paired, kind-labelled sidecar. "
                    f"Unset AUX_CE_KIND_BALANCE or point AUX_CE_DATA at the chain's final stage.")
                kinds[k] += 1
                block_cnt[(KIND_BLOCK[k], r["label"])] += 1
if kind_balance:
    # Each block's rows are matched pairs: the sampler draws one TRUE and one FALSE per pair.
    assert set(kinds) == set(KIND_BLOCK), (
        f"AUX_CE_KIND_BALANCE=1 needs all four kinds present, got {dict(kinds)}")
    for blk in ("a", "b"):
        t, f = block_cnt[(blk, "true")], block_cnt[(blk, "false")]
        assert t and f and t == f, (
            f"AUX_CE_KIND_BALANCE=1: block {blk!r} is label-imbalanced ({t}T/{f}F); its rows are "
            f"matched pairs, so both labels must be equinumerous")
assert have["true"] and have["false"], f"sidecar must hold BOTH labels (T={len(have['true'])} F={len(have['false'])} games)"
cov_t = len(have["true"] & train_ids) / len(train_ids)
cov_f = len(have["false"] & train_ids) / len(train_ids)
ratio = cnt["true"] / cnt["false"]
assert abs(ratio - 1.0) <= 0.01, (
    f"AUX_CE_SAMPLING=random needs a label-BALANCED sidecar, got {cnt['true']}T/{cnt['false']}F "
    f"(T:F={ratio:.3f}:1). Unequal pools mean unequal per-item exposure; rebuild with "
    f"datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py, which matches TRUE to FALSE 1:1.")
meta_path = Path(sidecar).parent.parent / "_meta.json"
meta = json.loads(meta_path.read_text())["sizes"][Path(sidecar).parent.name]
assert meta["max_prompt_tokens"] <= max_tok, f"sidecar max prompt {meta['max_prompt_tokens']} > AUX_CE_MAX_PROMPT {max_tok}"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(policy)
sha = hashlib.sha256(tok.chat_template.encode()).hexdigest()
assert sha == meta["chat_template_sha"], f"chat_template sha mismatch: policy {sha[:16]} != sidecar {meta['chat_template_sha'][:16]}"
print(f"[aux-ce] submit gate OK: {n} items ({cnt['true']}T/{cnt['false']}F, T:F={ratio:.3f}:1), "
      f"TRUE coverage {cov_t:.3f}, FALSE coverage {cov_f:.3f}, template sha match")
if kind_balance:
    print(f"[aux-ce] kind-balance gate OK: kinds={dict(sorted(kinds.items()))}, "
          f"blocks={{a: {block_cnt[('a','true')]}+{block_cnt[('a','false')]}, "
          f"b: {block_cnt[('b','true')]}+{block_cnt[('b','false')]}}}")
PYEOF
fi
# Mixture guards fail at job start, so a _mix-named run never silently trains 100% local.
MIX_SUFFIX=""
_MIX_META_MODEL=""; _MIX_META_FRAC=0.0; _MIX_META_SEED=0   # off state mirrors the yaml defaults
_MIX_META_SPEC=""; _MIX_META_ARMS=""; _MIX_META_WEIGHTS=""; _MIX_META_NARMS=0; _MIX_META_GATEWAY=""
_MIX_HOSTED=""; _MIX_PROBE_SPEC=""
RECV_MODEL_META="$JUDGE_MODEL"
if [[ -n "$RECEIVER_MIX_SPEC" && -n "$RECEIVER_MIX_MODEL" ]]; then
  echo "ERROR: RECEIVER_MIX_SPEC and RECEIVER_MIX_MODEL are mutually exclusive (N-way vs 2-way)."
  echo "       spec='$RECEIVER_MIX_SPEC'  model='$RECEIVER_MIX_MODEL'"
  echo "       For an N-way mixture put the local arm in the spec: RECEIVER_MIX_SPEC='local:4,<model>:4,...'"
  exit 1
fi
if [[ -n "$RECEIVER_MIX_SPEC" ]]; then
  # Validate with the runtime parser, not a bash copy that could drift. Any ValueError, including the
  # lone-'local' comma-truncation tripwire, fails the launch here.
  _MIX_JSON="$("$PY" - <<'PYEOF'
import json, os
import rl.receiver_mix as m
s = os.environ["RECEIVER_MIX_SPEC"]
arms = m.parse_spec(s)
print(json.dumps({
    "labels":  " ".join(lab for lab, _ in arms),
    "hosted":  " ".join(m.hosted_labels(arms)),
    "arms_csv":    ",".join(lab for lab, _ in arms),
    "weights_csv": ",".join(str(w) for _, w in arms),
    "n_arms":  len(arms),
    "suffix":  m.name_suffix(arms),
    "canon":   m.canonical(arms),
    "pretty":  " ".join(f"{lab}:{w}" for lab, w in arms),
}))
PYEOF
)" || {
    echo "ERROR: RECEIVER_MIX_SPEC is invalid (got '$RECEIVER_MIX_SPEC')."
    echo "       NOTE: 'sbatch --export' is ITSELF comma-separated, so an inline comma spec is"
    echo "       silently truncated to its first arm. Export it in your shell and pass a bare"
    echo "       --export=ALL, or use '+' as the arm separator."
    exit 1; }
  _mixjq() { "$PY" -c "import json,sys;print(json.loads(sys.stdin.read())['$1'])" <<<"$_MIX_JSON"; }
  _MIX_HOSTED="$(_mixjq hosted)"
  _MIX_META_ARMS="$(_mixjq arms_csv)"; _MIX_META_WEIGHTS="$(_mixjq weights_csv)"
  _MIX_META_NARMS="$(_mixjq n_arms)";  MIX_SUFFIX="$(_mixjq suffix)"
  _MIX_META_SPEC="$(_mixjq canon)";    _MIX_PROBE_SPEC="$RECEIVER_MIX_SPEC"
  echo "[mix] arms: $(_mixjq pretty) (n_arms=${_MIX_META_NARMS})"
  echo "[mix] hosted arms: ${_MIX_HOSTED:-<none>}"
  [[ -n "$_MIX_HOSTED" ]] || { echo "ERROR: RECEIVER_MIX_SPEC has no hosted arm -- that is the mixture OFF; use RECEIVER_MIX_SPEC=''"; exit 1; }
fi
if [[ -n "$RECEIVER_MIX_MODEL" ]]; then
  # regex first: awk reads non-numeric strings as 0
  [[ "$RECEIVER_MIX_FRAC" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || { echo "ERROR: RECEIVER_MIX_FRAC must be a decimal number (got '$RECEIVER_MIX_FRAC')"; exit 1; }
  awk -v f="$RECEIVER_MIX_FRAC" 'BEGIN{exit !(f>=0 && f<=1)}' || { echo "ERROR: RECEIVER_MIX_FRAC must be in [0,1] (got '$RECEIVER_MIX_FRAC')"; exit 1; }
  _MIX_HOSTED="$RECEIVER_MIX_MODEL"
  _MIX_PROBE_SPEC="${RECEIVER_MIX_MODEL}:1"
fi
if [[ -n "$_MIX_HOSTED" ]]; then
  # RECEIVER_MIX_GATEWAY_URL=local: loop back to the co-served judge (smoke, or a second local
  # receiver; no key needed).
  if [[ "$RECEIVER_MIX_GATEWAY_URL" == local ]]; then
    RECEIVER_MIX_GATEWAY_URL="http://${RECEIVER_HOST}:${RECEIVER_PORT}/v1/chat/completions"
    export RECEIVER_MIX_GATEWAY_URL
    echo "[mix] loopback gateway: $RECEIVER_MIX_GATEWAY_URL"
  fi
  # The hosted arms default to the API gateway at GATEWAY_URL, which has no default of its own.
  _MIX_URL="${RECEIVER_MIX_GATEWAY_URL:-${GATEWAY_URL:-}}"
  [[ -n "$_MIX_URL" ]] || { echo "ERROR: a hosted mixture arm is configured but GATEWAY_URL is empty (export the API gateway's chat-completions URL, or set RECEIVER_MIX_GATEWAY_URL)"; exit 1; }
  _MIX_HOST="$(echo "$_MIX_URL" | sed -E 's#^[a-zA-Z]+://([^/:]+).*#\1#')"
  # Bearer key from the env var named by RECEIVER_MIX_KEY_ENV. Non-interactive shells can miss it, so
  # fall back to its export line in ~/.bashrc. Loopback endpoints need none (sglang ignores auth).
  _MIX_KEY_ENV="${RECEIVER_MIX_KEY_ENV:-GATEWAY_API_KEY}"; export RECEIVER_MIX_KEY_ENV="$_MIX_KEY_ENV"
  _MIX_KEY="${!_MIX_KEY_ENV:-}"
  if [[ -z "$_MIX_KEY" ]]; then
    eval "$(grep -m1 "^export ${_MIX_KEY_ENV}=" ~/.bashrc 2>/dev/null || true)"
    _MIX_KEY="${!_MIX_KEY_ENV:-}"
  fi
  if [[ "$_MIX_HOST" == 127.0.0.1 || "$_MIX_HOST" == localhost ]]; then
    _MIX_KEY="${_MIX_KEY:-None}"
  else
    [[ -n "$_MIX_KEY" ]] || { echo "ERROR: a hosted mixture arm is configured but \$${_MIX_KEY_ENV} is empty (export it, or keep its export line in ~/.bashrc)"; exit 1; }
    export "${_MIX_KEY_ENV}"
  fi
  # The stratified assignment lives in the patched fit() (patch 10H); fail on a stale checkout.
  _MIX_RT="$("$PY" -c 'import verl.trainer.ppo.ray_trainer as m; print(m.__file__)')"
  grep -q "persuasion-gym] (10H)" "$_MIX_RT" || { echo "ERROR: a mixture is configured but the verl checkout ($_MIX_RT) lacks patch 10H -- run: bash rl/patch_verl.sh"; exit 1; }
  # An explicit RECEIVER_MIX_GATEWAY_URL's host is added to NO_PROXY. The preflight posts with
  # requests, like the runtime, so both take the same route.
  if [[ -n "$RECEIVER_MIX_GATEWAY_URL" ]]; then
    export NO_PROXY="$NO_PROXY,$_MIX_HOST"; export no_proxy="$NO_PROXY"
  fi
  # One preflight per hosted arm, with the body from rl.receiver_client.api_body() as at runtime. A
  # hand-rolled body without top_p passes while every runtime call gets HTTP 400, training a score-0
  # arm. The probe also requires non-empty content (reasoning models can return empty text).
  "$PY" -m rl.check_receiver_mix --live --spec "$_MIX_PROBE_SPEC" --url "$_MIX_URL" || {
    echo "ERROR: mixture preflight FAILED at $_MIX_URL."
    echo "       Check that this node can reach the endpoint and that \$${_MIX_KEY_ENV} is valid, or"
    echo "       point RECEIVER_MIX_GATEWAY_URL (+RECEIVER_MIX_KEY_ENV) at a reachable endpoint."
    exit 1; }
  _MIX_META_GATEWAY="$_MIX_URL"
  if [[ -n "$RECEIVER_MIX_SPEC" ]]; then
    # N-way '_mix<N>-<arms>-<hash6>' never collides with 2-way '_mix-<slug><frac>', so they resume
    # into separate dirs.
    RECV_MODEL_META="${JUDGE_MODEL}+$(echo "$_MIX_HOSTED" | tr ' ' '+')@$(echo "$_MIX_META_WEIGHTS" | tr ',' ':')"
    _MIX_META_SEED="$RECEIVER_MIX_SEED"
  else
    _MIX_SLUG="$(echo "$RECEIVER_MIX_MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
    MIX_SUFFIX="_mix-${_MIX_SLUG}${RECEIVER_MIX_FRAC//./p}"
    _MIX_META_MODEL="$RECEIVER_MIX_MODEL"; _MIX_META_FRAC="$RECEIVER_MIX_FRAC"; _MIX_META_SEED="$RECEIVER_MIX_SEED"
    RECV_MODEL_META="${JUDGE_MODEL}+${RECEIVER_MIX_MODEL}@${RECEIVER_MIX_FRAC}"
  fi
fi
# Run/checkpoint name: sender_{policy}_{neutral|stubborn}_{judge tag}_{penalty tag} plus the optional
# suffixes below. Every dimension that distinguishes a run is in the name, so auto-resume never loads
# the wrong FSDP checkpoint. SAVE_DIR, RL_JUDGE_DUMP_DIR and RL_EXP_NAME derive from it.
case "$DIST" in
  bayesian)  RECV_LABEL=neutral ;;
  stubborn)  RECV_LABEL=stubborn ;;
  *)         RECV_LABEL="$DIST" ;;
esac
case "$JUDGE_MODEL" in
  qwen3.5-35B)  JUDGE_TAG=j35B ;;
  *)            JUDGE_TAG="j_${JUDGE_MODEL}" ;;   # custom judge: still uniquely tagged
esac
# Penalty tag: fakepen0 (off), fakepen<lambda> (fabrication alone; '.'->'p'), pen<lambda>_illegal_full11
# (11 slugs would exceed NAME_MAX), else pen<lambda>_<sorted term abbrevs> (e.g. pen0p1_belief_fab).
_ILLEGAL_SORTED="$(echo $_ILLEGAL_TERMS | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/^ *//; s/ *$//')"
if [[ "$_coeff_zero" == 1 || -z "$EFF_TERMS" ]]; then
  FAKEPEN_TAG=fakepen0
elif [[ "$EFF_TERMS" == "fabrication" ]]; then
  FAKEPEN_TAG="fakepen${EFF_COEFF//./p}"
elif [[ "$EFF_TERMS" == "$_ILLEGAL_SORTED" ]]; then
  FAKEPEN_TAG="pen${EFF_COEFF//./p}_illegal_full11"
else
  _abbr=""; for _t in $EFF_TERMS; do case "$_t" in
    fabrication)        _abbr="${_abbr}fab_" ;;
    fabrication_binary) _abbr="${_abbr}fabbin_" ;;
    belief)             _abbr="${_abbr}belief_" ;;
    *)                  _abbr="${_abbr}${_t}_" ;;
  esac; done
  FAKEPEN_TAG="pen${EFF_COEFF//./p}_${_abbr%_}"
fi
# _fmt<coeff> when the format bonus is on. Coeff 0 changes the reward and name only, not the sender
# prompt, so never reuse another arm's EXP_NAME with resume enabled.
FMT_SUFFIX=""
case "$FORMAT_REWARD_COEFF" in ""|0) : ;; *) FMT_SUFFIX="_fmt${FORMAT_REWARD_COEFF//./p}" ;; esac
# _rs<target>[r<rounds> if != 2]: changes which rollouts train. Uses the configured target; the
# effective min(target, N_ROLLOUT) is logged as rejection/target_clean_effective.
RS_SUFFIX=""
_RS_META_TARGET=0; _RS_META_ROUNDS=0   # off state mirrors grpo_persuasion.yaml's 0/0/0 defaults
if [[ "${REJECTION_SAMPLING}" == 1 ]]; then
  RS_SUFFIX="_rs${RS_TARGET_CLEAN}"
  [[ "$RS_MAX_GEN_ROUNDS" != 2 ]] && RS_SUFFIX="${RS_SUFFIX}r${RS_MAX_GEN_ROUNDS}"
  _RS_META_TARGET="$RS_TARGET_CLEAN"; _RS_META_ROUNDS="$RS_MAX_GEN_ROUNDS"
fi
# Prompt, _sftsplit and _sftinit[-tag] suffixes each change the trained checkpoint, so none can
# auto-resume a from-base 1121-game run of the same name.
SFTSPLIT_SUFFIX=""; [[ "$SFT_SPLIT" == 1 ]] && SFTSPLIT_SUFFIX="_sftsplit"
SFTINIT_SUFFIX=""; [[ -n "$SFT_INIT" ]] && SFTINIT_SUFFIX="_sftinit${SFT_INIT_TAG:+-$SFT_INIT_TAG}"
# _auxce<coeff>[f<frac>][_<sampling>][_kindbal][_normratio<target>]. AUX_CE_DATA is not encoded, so a
# sidecar ablation must pass an explicit EXP_NAME or resume_mode=auto resumes the other run.
AUXCE_SUFFIX=""
_AUXCE_META_COEFF=0.0; _AUXCE_META_FRAC=0.0; _AUXCE_META_SAMPLING=none   # off state
_AUXCE_META_BALANCE_MODE=off; _AUXCE_META_TARGET_RATIO=0.0
_AUXCE_META_KIND_BALANCE=0
AUXBAL_SUFFIX=""; AUXKIND_SUFFIX=""
if [[ "$AUX_CE" == 1 ]]; then
  AUXCE_SUFFIX="_auxce${AUX_CE_COEFF//./p}"
  [[ "$AUX_CE_ROW_FRAC" != 1.0 && "$AUX_CE_ROW_FRAC" != 1 ]] && AUXCE_SUFFIX="${AUXCE_SUFFIX}f${AUX_CE_ROW_FRAC//./p}"
  [[ "$AUX_CE_SAMPLING" != game ]] && AUXCE_SUFFIX="${AUXCE_SUFFIX}_${AUX_CE_SAMPLING}"
  # Kind balance changes which items the aux batch holds, so it goes in the name.
  [[ "$AUX_CE_KIND_BALANCE" == 1 ]] && AUXKIND_SUFFIX="_kindbal" && \
    AUXCE_SUFFIX="${AUXCE_SUFFIX}${AUXKIND_SUFFIX}"
  if [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]]; then
    AUXBAL_SUFFIX="_normratio${AUX_CE_TARGET_GRAD_RATIO//./p}"
    AUXCE_SUFFIX="${AUXCE_SUFFIX}${AUXBAL_SUFFIX}"
  fi
  _AUXCE_META_COEFF="$AUX_CE_COEFF"; _AUXCE_META_FRAC="$AUX_CE_ROW_FRAC"
  _AUXCE_META_SAMPLING="$AUX_CE_SAMPLING"
  _AUXCE_META_BALANCE_MODE="$AUX_CE_BALANCE_MODE"
  _AUXCE_META_KIND_BALANCE="$AUX_CE_KIND_BALANCE"
  [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]] && _AUXCE_META_TARGET_RATIO="$AUX_CE_TARGET_GRAD_RATIO"
fi
RSEED_SUFFIX=""; [[ -n "$ROLLOUT_SEED" ]] && RSEED_SUFFIX="_rseed${ROLLOUT_SEED}"
_EXP_NAME_EXPLICIT="${EXP_NAME:-}"
EXP_NAME="${EXP_NAME:-sender_${POLICY_MODEL}_${RECV_LABEL}_${JUDGE_TAG}_${FAKEPEN_TAG}${FMT_SUFFIX}${RS_SUFFIX}${PROMPT_SUFFIX}${SFTSPLIT_SUFFIX}${SFTINIT_SUFFIX}${AUXCE_SUFFIX}${MIX_SUFFIX}${RSEED_SUFFIX}${SMOKE_SUFFIX}}"
# SAVE_DIR relocates the durable results; the default is repository-local.
SAVE_DIR="${SAVE_DIR:-$REPO/experiments/results/rl/$EXP_NAME}"
if [[ -n "$ROLLOUT_SEED" ]]; then
  export ROLLOUT_SEED_PROVENANCE_DIR="${ROLLOUT_SEED_PROVENANCE_DIR:-$SAVE_DIR/rollout_seed_provenance}"
else
  unset ROLLOUT_SEED_PROVENANCE_DIR
fi
# Seeded sampling changes the training data: an explicit name must carry the exact _rseed tag and an
# unseeded run must not carry one, so resume_mode=auto cannot cross that boundary.
if [[ -n "$ROLLOUT_SEED" && ! "$EXP_NAME" =~ (^|_)rseed${ROLLOUT_SEED}(_|$) ]]; then
  echo "ERROR: explicit EXP_NAME='$EXP_NAME' does not contain rollout-seed tag '$RSEED_SUFFIX'."
  exit 1
fi
if [[ -z "$ROLLOUT_SEED" && "$EXP_NAME" =~ (^|_)rseed[0-9]+(_|$) ]]; then
  echo "ERROR: EXP_NAME='$EXP_NAME' advertises deterministic rollout sampling but ROLLOUT_SEED is unset."
  exit 1
fi
# An explicit EXP_NAME (the chain scripts pass one) must still contain the mixture tag: resume_mode=auto
# validates nothing, so a later leg could continue a checkpoint under a different mixture.
if [[ -n "$_EXP_NAME_EXPLICIT" && -n "$MIX_SUFFIX" && "$EXP_NAME" != *"$MIX_SUFFIX"* ]]; then
  echo "ERROR: explicit EXP_NAME='$EXP_NAME' does not contain the mixture tag '$MIX_SUFFIX'."
  echo "       Refusing to write a run whose name misdescribes its receiver mixture."
  exit 1
fi
# Same for adaptive aux balancing: it changes every update, so the name must carry _normratio.
if [[ -n "$_EXP_NAME_EXPLICIT" && -n "$AUXBAL_SUFFIX" && "$EXP_NAME" != *"$AUXBAL_SUFFIX"* ]]; then
  echo "ERROR: explicit EXP_NAME='$EXP_NAME' does not contain the aux-balance tag '$AUXBAL_SUFFIX'."
  echo "       Refusing to write an adaptive run whose name omits its target gradient ratio."
  exit 1
fi
if [[ "$AUX_CE_BALANCE_MODE" == off && "$EXP_NAME" == *_normratio* ]]; then
  echo "ERROR: EXP_NAME='$EXP_NAME' advertises norm-ratio aux balancing but AUX_CE_BALANCE_MODE=off."
  exit 1
fi
# Same for kind balance: balanced and unbalanced twins must not share a checkpoint dir.
if [[ -n "$_EXP_NAME_EXPLICIT" && -n "$AUXKIND_SUFFIX" && "$EXP_NAME" != *"$AUXKIND_SUFFIX"* ]]; then
  echo "ERROR: explicit EXP_NAME='$EXP_NAME' does not contain the kind-balance tag '$AUXKIND_SUFFIX'."
  echo "       Refusing to write a kind-balanced run whose name omits it."
  exit 1
fi
if [[ "$AUX_CE_KIND_BALANCE" != 1 && "$EXP_NAME" == *_kindbal* ]]; then
  echo "ERROR: EXP_NAME='$EXP_NAME' advertises per-batch kind balance but AUX_CE_KIND_BALANCE!=1."
  exit 1
fi
# The converse: a _mix name with no mixture configured (e.g. the spec was lost to --export comma
# splitting) would train 100% local into the mixture's checkpoint dir.
if [[ -z "$MIX_SUFFIX" && "$EXP_NAME" == *_mix* ]]; then
  echo "ERROR: EXP_NAME='$EXP_NAME' advertises a receiver mixture but none is configured"
  echo "       (RECEIVER_MIX_SPEC='${RECEIVER_MIX_SPEC:-}' RECEIVER_MIX_MODEL='${RECEIVER_MIX_MODEL:-}')."
  echo "       Refusing to train 100% local into a mixture-named run. NOTE 'sbatch --export' is"
  echo "       itself comma-separated: export RECEIVER_MIX_SPEC in your shell and pass --export=ALL."
  exit 1
fi
# Announce auto-resume. find, not an ls glob: with no checkpoints ls exits 2 and pipefail + set -e
# abort the job silently. SAVE_DIR can exist without checkpoints (the MIXTURE.txt mkdir below).
if [[ -d "$SAVE_DIR" ]]; then
  _EXIST_CKPT="$(find "$SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | wc -l)"
  if [[ "$_EXIST_CKPT" != 0 ]]; then
    echo "[mix] NOTE: $SAVE_DIR already has $_EXIST_CKPT checkpoint(s) -- this run will AUTO-RESUME (latest=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?'))"
  fi
fi
# Mixture stamp: written on the first leg, verified on later legs so a resume cannot change the mixture.
if [[ -n "$MIX_SUFFIX" ]]; then
  mkdir -p "$SAVE_DIR"
  _MIX_STAMP_WANT="${_MIX_META_SPEC:-legacy:${RECEIVER_MIX_MODEL}@${RECEIVER_MIX_FRAC}}"
  if [[ -f "$SAVE_DIR/MIXTURE.txt" ]]; then
    _MIX_STAMP_HAVE="$(cat "$SAVE_DIR/MIXTURE.txt")"
    [[ "$_MIX_STAMP_HAVE" == "$_MIX_STAMP_WANT" ]] || {
      echo "ERROR: $SAVE_DIR was trained with mixture '$_MIX_STAMP_HAVE' but this job configures"
      echo "       '$_MIX_STAMP_WANT'. Refusing to resume a differently-configured run."; exit 1; }
  else
    printf '%s' "$_MIX_STAMP_WANT" > "$SAVE_DIR/MIXTURE.txt"
  fi
fi
# Raw-judge dump (rl/monitors.py calls), per run under $SAVE_DIR/judges (patch_verl.sh 10D/10E/10F):
# judges_step_NNNNN.yaml (train, 1:1 with rollouts/<step>.jsonl) and judges_val_step_NNNNN.yaml.
# Records carry {step, phase, exp, game_id, call_type, messages, response}.
export RL_JUDGE_DUMP_DIR="${RL_JUDGE_DUMP_DIR:-$SAVE_DIR/judges}"
export RL_EXP_NAME="$EXP_NAME"

# Full-parameter FT with params + optimizer on GPU, constant LR without warmup (lr itself is in
# grpo_persuasion.yaml). EXTRA_OVERRIDES comes last so it wins.
echo "[train] DIST=$DIST exp=$EXP_NAME data=$_DATA_VARIANT/$DIST batch=$TRAIN_BATCH n=$N_ROLLOUT steps=$TOTAL_STEPS gpus=$TRAINER_GPUS rollout_tp=$ROLLOUT_TP rollout_seed=${ROLLOUT_SEED:-off} aux_balance=$_AUXCE_META_BALANCE_MODE aux_target=$_AUXCE_META_TARGET_RATIO"
CUDA_VISIBLE_DEVICES=$TRAINER_GPUS "$PY" -m verl.trainer.main_ppo \
  --config-path "$REPO/rl/config" --config-name grpo_persuasion \
  data.train_files="$DATA_DIR/rl_train.parquet" \
  data.val_files="$DATA_DIR/rl_validation.parquet" \
  data.train_batch_size="$TRAIN_BATCH" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH" \
  actor_rollout_ref.model.path="$POLICY_PATH" \
  actor_rollout_ref.rollout.n="$N_ROLLOUT" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MEM_FRAC" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.rollout_data_dir="$SAVE_DIR/rollouts" \
  custom_metadata.sender_model="$POLICY_MODEL" \
  custom_metadata.receiver_model="$RECV_MODEL_META" \
  custom_metadata.receiver_mix_model="'$_MIX_META_MODEL'" \
  custom_metadata.receiver_mix_frac="$_MIX_META_FRAC" \
  custom_metadata.receiver_mix_seed="$_MIX_META_SEED" \
  custom_metadata.receiver_mix_spec="'$_MIX_META_SPEC'" \
  custom_metadata.receiver_mix_arms="'$_MIX_META_ARMS'" \
  custom_metadata.receiver_mix_weights="'$_MIX_META_WEIGHTS'" \
  custom_metadata.receiver_mix_n_arms="$_MIX_META_NARMS" \
  custom_metadata.receiver_mix_gateway="'$_MIX_META_GATEWAY'" \
  custom_metadata.judge_model="$JUDGE_MODEL" \
  custom_metadata.served_model_path="$JUDGE_PATH" \
  custom_metadata.receiver_distribution="$DIST" \
  custom_metadata.receiver_types="$RECEIVER_TYPES" \
  custom_metadata.receiver_prior="$RECEIVER_PRIOR" \
  custom_metadata.rollout_seed="$_ROLLOUT_SEED_META" \
  custom_metadata.rollout_deterministic_inference="$_ROLLOUT_DETERMINISTIC_META" \
  custom_metadata.rollout_seed_impl="$_ROLLOUT_SEED_IMPL_META" \
  trainer.project_name="$WANDB_PROJECT" \
  custom_metadata.penalty_terms="'$(echo $EFF_TERMS | tr ' ' ',')'" \
  custom_metadata.penalty_coefficient="$EFF_COEFF" \
  custom_metadata.format_reward_coeff="$FORMAT_REWARD_COEFF" \
  custom_metadata.rejection_sampling="$REJECTION_SAMPLING" \
  custom_metadata.rs_target_clean="$_RS_META_TARGET" \
  custom_metadata.rs_max_gen_rounds="$_RS_META_ROUNDS" \
  custom_metadata.sender_prompt="$SENDER_PROMPT_TAG" \
  custom_metadata.algo="'$ALGO_RESOLVED'" \
  custom_metadata.prompt="'$PROMPT'" \
  custom_metadata.dataset_name="$_DATA_VARIANT/$DIST" \
  custom_metadata.policy_init="${SFT_INIT:-base}" \
  custom_metadata.aux_ce="$AUX_CE" \
  custom_metadata.aux_ce_coeff="$_AUXCE_META_COEFF" \
  custom_metadata.aux_ce_row_frac="$_AUXCE_META_FRAC" \
  custom_metadata.aux_ce_sampling="$_AUXCE_META_SAMPLING" \
  custom_metadata.aux_ce_balance_mode="'$_AUXCE_META_BALANCE_MODE'" \
  custom_metadata.aux_ce_kind_balance="$_AUXCE_META_KIND_BALANCE" \
  custom_metadata.aux_ce_target_grad_ratio="$_AUXCE_META_TARGET_RATIO" \
  custom_metadata.aux_ce_data="'$AUX_CE_DATA'" \
  custom_metadata.teacher_model="$TEACHER_MODEL" \
  ${SMOKE_OVERRIDES} \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
  actor_rollout_ref.actor.optim.num_cycles=0.5 \
  ${EXTRA_OVERRIDES:-}

echo "[done] $EXP_NAME -> $SAVE_DIR. Offline wandb under $REPO/wandb; upload it with wandb sync."
