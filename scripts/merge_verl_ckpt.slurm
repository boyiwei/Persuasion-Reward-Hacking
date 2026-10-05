#!/bin/bash
#SBATCH --job-name=merge_verl
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Merge verl FSDP actor checkpoints into single HF dirs so SGLang can serve them as the model under
# test (EM eval and the {oldbailey,houseshowing,nutrition}_rl_eval drivers). CPU-bound
# (map_location=cpu). Existing outputs are skipped.
#
# Every source is under RL_SRC, the run dirs scripts/rl_train_sender.slurm writes (default
# <repo>/experiments/results/rl, where <repo> is the submit dir). OUTROOT sets the merged root
# (default <repo>/experiments/results/rl/_em_merged); ONLY=<substr> merges only the matching output
# slugs. To merge another checkout's runs, point RL_SRC at its results:
#   sbatch --export=ALL,RL_SRC=/path/to/checkout/experiments/results/rl scripts/merge_verl_ckpt.slurm
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
PY="${MERGE_PY:-$REPO/.venv/bin/python}"
# Without a python, every merge fails, the skip handler swallows it, and the job exits 0 having merged
# nothing. A checkout without its own .venv must pass MERGE_PY.
[ -x "$PY" ] || { echo "[merge] ERROR: no python at '$PY' -- set MERGE_PY (e.g. \$MAIN/.venv/bin/python)"; exit 1; }
_MAIN_RL="$REPO/experiments/results/rl"
RL_SRC="${RL_SRC:-$_MAIN_RL}"          # source root of every entry below
OUTROOT="${OUTROOT:-$_MAIN_RL/_em_merged}"
mkdir -p "$OUTROOT"

# actor checkpoint dir  ->  output HF slug
declare -A CKPTS=(
  # --- neutral full-FT (j35B) fakepen0 fmt0p1 gs100: the RL-neutral arm.
  #     ONLY=neutral_j35B_fakepen0_fmt0p1 ---
  ["$RL_SRC/sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1/global_step_100/actor"]="sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs100"
  ["$RL_SRC/sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1/global_step_100/actor"]="sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1_gs100"
  ["$RL_SRC/sender_qwen3-14B_neutral_j35B_fakepen0_fmt0p1/global_step_100/actor"]="sender_qwen3-14B_neutral_j35B_fakepen0_fmt0p1_gs100"
  # gs50 of the 4B/8B neutral arms, pairing the stubborn gs50 entries below (steps 0/50/100 for both
  # juror profiles). ONLY=qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs; keep the `qwen3-` prefix, since
  # `4B_neutral…` also matches the 14B slug.
  ["$RL_SRC/sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1/global_step_50/actor"]="sender_qwen3-4B_neutral_j35B_fakepen0_fmt0p1_gs50"
  ["$RL_SRC/sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1/global_step_50/actor"]="sender_qwen3-8B_neutral_j35B_fakepen0_fmt0p1_gs50"
  ["$RL_SRC/sender_qwen3-8B_neutral_j35B_fakepen0/global_step_100/actor"]="sender_qwen3-8B_neutral_j35B_fakepen0_gs100"
  ["$RL_SRC/sender_qwen3-4B_neutral_j35B_fakepen0/global_step_100/actor"]="sender_qwen3-4B_neutral_j35B_fakepen0_gs100"
  # --- stubborn full-FT (j35B) gs100 — canonical _argfix runs ---
  ["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_argfix/global_step_100/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_argfix_gs100"
  ["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_argfix/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_argfix_gs100"
  ["$RL_SRC/sender_qwen3-14B_stubborn_j35B_fakepen0_argfix/global_step_100/actor"]="sender_qwen3-14B_stubborn_j35B_fakepen0_argfix_gs100"
  # --- stubborn full-FT (j35B) fakepen0 fmt0p1, gs50 + gs100. Base sender prompt: evaluate with
  #     PROMPT=base, or the eval measures the prompt rather than the checkpoint.
  #     ONLY=stubborn_j35B_fakepen0_fmt0p1_gs; the trailing `_gs` excludes the
  #     _strategies_sftsplit_* descendants. ---
  ["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1/global_step_50/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs50"
  ["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_gs100"
  ["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1/global_step_50/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs50"
  ["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1/global_step_100/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_gs100"
)
# --- SFT-then-RL distill: 921-game sftsplit from-base baselines + SFT-initialized arms.
#     ONLY=misrepv2 ---
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs100"
CKPTS["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2/global_step_100/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs100"
# gs50 of the two from-base arms: mid-trajectory point, and the prober the aux-CE probe sidecar is
# built from. ONLY=misrepv2_gs50
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2/global_step_50/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs50"
CKPTS["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2/global_step_50/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_gs50"
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2_gs100"
CKPTS["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2/global_step_100/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_sftinit-ep2_gs100"
# LR ablation (4B from-base, lr 3e-6 vs the default 1e-6, 200 steps). gs100 matches the 100-step main
# arms; gs200 is its full run.
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6_gs100"
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6/global_step_200/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_lr3e-6_gs200"
# Self-distillation arms (teacher == student Qwen3-4B/8B), SFT-initialized on the same 921-game data.
# ONLY=selfdistill_sftinit
CKPTS["$RL_SRC/sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2/global_step_100/actor"]="sender_qwen3-4B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2_gs100"
CKPTS["$RL_SRC/sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2/global_step_100/actor"]="sender_qwen3-8B_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_selfdistill_sftinit-ep2_gs100"

# --- aux-CE runs (deception-probe CE loss, AUX_CE=1, λ=0.5). ONLY=auxce0p5 ---
for _SZ in 4B 8B; do
  _RUN="sender_qwen3-${_SZ}_stubborn_j35B_fakepen0_fmt0p1_strategies_sftsplit_misrepv2_auxce0p5"
  for _GS in 50 100; do
    CKPTS["$RL_SRC/$_RUN/global_step_${_GS}/actor"]="${_RUN}_gs${_GS}"
  done
done

# --- illegal_full11 penalty grid: 11 binary illegal-strategy penalties at lambda=0.3, 921-game
#     rl_sftsplit_strategies data, stubborn juror, fmt0p1, x 3 GRPO inits (from-base, gpt-5.4-mini
#     SFT, self-distill SFT). 200 steps; gs100/150/200 merged. ONLY=illegal_full11 ---
for _SZ in 4B 8B; do
  for _ARM in _misrepv2 _misrepv2_sftinit-ep2 _sftinit-selfdistill-ep2; do
    _RUN="sender_qwen3-${_SZ}_stubborn_j35B_pen0p3_illegal_full11_fmt0p1_strategies_sftsplit${_ARM}"
    for _GS in 100 150 200; do
      CKPTS["$RL_SRC/$_RUN/global_step_${_GS}/actor"]="${_RUN}_gs${_GS}"
    done
  done
done

echo "[merge] verl: $("$PY" -c 'import verl;print(verl.__version__)')"
for SRC in "${!CKPTS[@]}"; do
  OUT="$OUTROOT/${CKPTS[$SRC]}"
  # Optional one-off filter: ONLY=<substr> merges just the slugs containing <substr>.
  if [ -n "${ONLY:-}" ] && [[ "${CKPTS[$SRC]}" != *"$ONLY"* ]]; then continue; fi
  echo "=============================================================="
  echo "[merge] $SRC -> $OUT"
  if [ -f "$OUT/config.json" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
    echo "[merge] SKIP (already merged): $OUT"; continue
  fi
  # Skip a checkpoint that does not exist yet, so ready ones (e.g. gs100) merge while gs200 trains.
  # STRICT_MERGE=1 hard-fails.
  if [ ! -d "$SRC" ]; then
    if [ "${STRICT_MERGE:-0}" = 1 ]; then echo "[merge] ERROR: missing $SRC"; exit 1; fi
    echo "[merge] SKIP (source not present yet): $SRC"; continue
  fi
  # An interrupted save lacks huggingface/config.json, and the merger then crashes with a misleading
  # "Repo id must be..." OSError. Skip it (STRICT_MERGE=1 hard-fails).
  if [ ! -f "$SRC/huggingface/config.json" ]; then
    if [ "${STRICT_MERGE:-0}" = 1 ]; then echo "[merge] ERROR: $SRC/huggingface/config.json missing"; exit 1; fi
    echo "[merge] SKIP (no huggingface/config.json — reconstruct from a sibling global_step): $SRC"; continue
  fi
  # A merge crash must not abort the remaining entries; drop the partial OUT and move on.
  if ! "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$OUT"; then
    echo "[merge] ERROR merging $SRC -> $OUT (skipping; removing partial dir)"; rm -rf "$OUT"
    _N_ERR=$((${_N_ERR:-0}+1)); continue
  fi
  # SGLang chat endpoint needs the tokenizer + chat_template; copy any the merger didn't emit.
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json vocab.json merges.txt special_tokens_map.json; do
    [ -f "$SRC/huggingface/$f" ] && [ ! -f "$OUT/$f" ] && cp "$SRC/huggingface/$f" "$OUT/" && echo "[merge]   copied $f"
  done
  echo "[merge] done: $(ls -1 "$OUT" | tr '\n' ' ')"
  _N_OK=$((${_N_OK:-0}+1))
done
# Exit nonzero if every attempted merge failed: skip-and-continue protects healthy checkpoints, not
# a wholesale failure.
_N_OK="${_N_OK:-0}"; _N_ERR="${_N_ERR:-0}"
echo "[merge] merged $_N_OK, failed $_N_ERR, skipped-existing ${_N_SKIP:-0} -> $OUTROOT"
if [ "$_N_OK" -eq 0 ] && [ "$_N_ERR" -gt 0 ]; then
  echo "[merge] ERROR: every attempted merge failed -- exiting nonzero so this is not read as success"
  exit 1
fi
echo "[merge] ALL DONE -> $OUTROOT"
