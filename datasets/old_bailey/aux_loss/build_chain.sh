#!/bin/bash
# Build the aux-CE train sidecar `scripts/rl_train_sender.slurm` reads under AUX_CE=1.
#
# Stages, one directory each under <GEN>/aux_ce_probe/variants/, each a transform of the previous:
#   base_origin_all  every base-checkpoint fabrication, matched 1:1 by a TRUE record line from the
#                    same game (no bootstrapping from trained checkpoints)
#   base_origin      the 647-FALSE draw nested inside it, so a smaller dose is a subset of a larger
#   balanced         the verbatim TRUE record line replaced by a length- and style-matched rewrite,
#                    plus a block of paraphrase/altered hard negatives
#   kind_balanced    481 rows x {fabrication, claim_real, altered_real, paraphrase_real} = 1924 at
#                    both sizes, the shape AUX_CE_KIND_BALANCE=1 draws from
#   final            block-A TRUE set back to the record line it was written from; the paper's
#                    "Reference" row and the dataset training reads
#
# Step 1 inputs you must produce first (the FALSE class is what a sender actually said). For each
# policy size and each origin checkpoint (base / step 50 / step 100 of a strategies-prompt GRPO run):
#   * a train-split rollout    evaluation/rl_rollout.py --domain old-bailey --profile stubborn \
#                                  --sender-prompt strategies --split train
#   * its claim-level audit    evaluation/source_of_fabrication/audit_rollouts.py
# Point --report-dir at the audits (--audit-tmpl names them) and --results-json at the rollouts.
#
# Steps 1-2 and 4-6 are CPU only. Step 3 calls a hosted rewrite model, so it needs API access.
#
#   bash datasets/old_bailey/aux_loss/build_chain.sh                # every step
#   STEPS="4 5 6" bash datasets/old_bailey/aux_loss/build_chain.sh  # resume at the validation reference
#   DRY_RUN=1 bash datasets/old_bailey/aux_loss/build_chain.sh      # print the commands only
#
# Knobs: SIZES("4B 8B") STEPS("1 2 3 4 5 6") GEN(the generated root)
#        REPORT_DIR(step 1's audits; required whenever step 1 actually runs)
#        RESULTS_JSON(step 1's rollout map) TOK_4B/TOK_8B(tokenizers, default $MODELS_DIR/<Model>)
#        MODELS_DIR(<repo>/models) PY DRY_RUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
GEN="${GEN:-$REPO/datasets/old_bailey/_generated}"
SIZES="${SIZES:-4B 8B}"
STEPS="${STEPS:-1 2 3 4 5 6}"
DRY_RUN="${DRY_RUN:-0}"
# The tokenizers the sidecar contexts are token-checked with (rl/fab_aux_common.TOKENIZER_PATH).
TOK_4B="${TOK_4B:-${MODELS_DIR:-$REPO/models}/Qwen3-4B-Instruct-2507}"
TOK_8B="${TOK_8B:-${MODELS_DIR:-$REPO/models}/Qwen3-8B}"

PROBE="$GEN/aux_ce_probe"
VARIANTS="$PROBE/variants"
TEMPLATE="$REPO/datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json"
TRAIN_PARQUET="$GEN/rl_sftsplit_strategies/stubborn/rl_train.parquet"
SPLIT_FILE="$GEN/rl_sftsplit_strategies/sft_split.json"

run() {   # echo the command, then run it unless DRY_RUN=1
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY_RUN" = 1 ] || "$@"
}
step() { case " $STEPS " in *" $1 "*) return 0 ;; *) echo "[skip] step $1"; return 1 ;; esac; }
tok_for() { case "$1" in 4B) echo "$TOK_4B" ;; 8B) echo "$TOK_8B" ;; *) echo "[chain] unknown size $1" >&2; exit 1 ;; esac; }
# shellcheck disable=SC2206  # SIZES is intentionally word-split into the --sizes list
SIZE_ARGS=($SIZES)

# --- 1. the raw probe sidecar: one in-role deception-probe item per train game ------------------
# FALSE = a judge-confirmed fabrication spliced into the exact context that produced it;
# TRUE = a real record line, plus control_real_aug items until TRUE ~= FALSE.
if step 1; then
  # A dry run prints the argv with a placeholder instead of demanding the real directory.
  [ "$DRY_RUN" = 1 ] || : "${REPORT_DIR:?step 1 needs REPORT_DIR=<dir holding the claim-level audits> (see the header)}"
  run "$PY" "$HERE/build_aux_ce_sidecar.py" \
    --report-dir "${REPORT_DIR:-/path/to/claim-level-audits}" \
    --sizes "${SIZE_ARGS[@]}" \
    --sender-prompt strategies \
    --out-dir "$PROBE" \
    --split-file "$SPLIT_FILE" \
    ${RESULTS_JSON:+--results-json "$RESULTS_JSON"}
fi

# --- 2. base_origin_all -> base_origin: base FALSE only, TRUE re-matched 1:1, nested dose levels -
if step 2; then
  run "$PY" "$HERE/filter_aux_ce_sidecar.py" --tag base_origin_all --false-origin base \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --out-root "$VARIANTS" \
    --train-parquet "$TRAIN_PARQUET" --selfcheck
  run "$PY" "$HERE/filter_aux_ce_sidecar.py" --tag base_origin --false-origin base \
    --n-false 647 --n-false-8B 647 --nest-in base_origin_all \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --out-root "$VARIANTS" \
    --train-parquet "$TRAIN_PARQUET" --selfcheck
fi

# --- 3. balanced: kill the surface-form shortcut ----------------------------------------------
# --model (gpt-5.4-mini) rewrites each TRUE record line to match a FALSE partner's word count and
# style, and adds the paraphrase/altered hard-negative block. The stock `openai` client uses
# OPENAI_API_KEY and OPENAI_BASE_URL (the gateway serving --model). --cache makes a re-run free and
# byte-identical. The balance gate must pass here.
if step 3; then
  run "$PY" "$HERE/rebalance_aux_ce_sidecar.py" --parent base_origin --tag balanced \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --out "$VARIANTS" \
    --template "$TEMPLATE" --cache "$GEN/cache/aux_ce_rebalance_cache.json" \
    --hard-cap-per-game 4 --claim-word-tol 1 --attempts 4 --workers 16 \
    --temperature 0 --retry-temperature 0.7 --seed 0
  run "$PY" "$HERE/check_statement_balance.py" --variant balanced \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --template "$TEMPLATE"
fi

# --- 4. the frozen validation reference ---------------------------------------------------------
# Every later stage's val split is checked against this corpus's 30-game val holdout
# (make_train_sidecar_meta proof #5), so it is built before any trimmed stage.
if step 4; then
  for SZ in "${SIZE_ARGS[@]}"; do
    run "$PY" "$HERE/build_auxce_sft_dataset.py" \
      --sidecar "$VARIANTS/balanced/$SZ/sidecar.jsonl.gz" \
      --meta "$VARIANTS/balanced/_meta.json" \
      --size "$SZ" --tokenizer "$(tok_for "$SZ")" \
      --out-dir "$GEN/sft/auxce_balanced_$SZ" --max-len 7168
  done
fi

# --- 5. kind_balanced: 481 pairs per block, dumped back out as a train-only sidecar -------------
# The trim equalises both blocks and both sizes (1924 items = 4 kinds x 481), the shape
# AUX_CE_KIND_BALANCE=1 requires. make_train_sidecar_meta re-proves it and writes the _meta.json
# the loader and the submit gate read.
if step 5; then
  for SZ in "${SIZE_ARGS[@]}"; do
    run "$PY" "$HERE/build_auxce_sft_dataset.py" \
      --sidecar "$VARIANTS/balanced/$SZ/sidecar.jsonl.gz" \
      --meta "$VARIANTS/balanced/_meta.json" \
      --size "$SZ" --tokenizer "$(tok_for "$SZ")" \
      --out-dir "$GEN/sft/auxce_kind_balanced_$SZ" --max-len 7168 \
      --train-pairs-a 481 --train-pairs-b 481 --pair-seed 2026 \
      --dump-train-sidecar "$VARIANTS/kind_balanced/$SZ/sidecar.jsonl.gz"
  done
  run "$PY" "$HERE/make_train_sidecar_meta.py" --tag kind_balanced --parent balanced \
    --root "$VARIANTS" --sizes "${SIZE_ARGS[@]}" \
    --sft-dir-tpl "$GEN/sft/auxce_kind_balanced_{size}" \
    --ref-sft-dir-tpl "$GEN/sft/auxce_balanced_{size}" \
    --expect-items 1924 --expect-pairs-a 481 --expect-pairs-b 481
fi

# --- 6. final: set claim_real to the verbatim record line ---------------------------------------
# No model, no seed: every other row is byte-identical to kind_balanced. The balance gate fails
# here by design (block A is a long verbatim TRUE against a short fabrication); --expect-fail
# records that confound as a number instead of relaxing the gate.
if step 6; then
  run "$PY" "$HERE/verbatim_claim_sidecar.py" --parent kind_balanced --tag final \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --out "$VARIANTS" --template "$TEMPLATE" \
    --sft-dir-tpl "$GEN/sft/auxce_kind_balanced_{size}" \
    --val-sft-dir-tpl "$GEN/sft/auxce_balanced_{size}" --selfcheck
  run "$PY" "$HERE/check_statement_balance.py" --variant final --expect-fail \
    --sizes "${SIZE_ARGS[@]}" --src "$PROBE" --template "$TEMPLATE"
fi

echo "[chain] done. AUX_CE_DATA for each size:"
for SZ in "${SIZE_ARGS[@]}"; do
  echo "  $VARIANTS/final/$SZ/sidecar.jsonl.gz"
done
