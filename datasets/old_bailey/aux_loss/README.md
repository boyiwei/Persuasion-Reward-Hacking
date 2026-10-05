# The auxiliary-loss dataset

The fabrication-detection set the auxiliary cross-entropy loss trains on. Under `ALGO=base+aux_loss`,
every GRPO optimizer step adds a CE loss on the gold `true`/`false` answer token of one probe item per
rollout row, drawn from this sidecar. Each item is an in-role chat: the 42-technique guide system
prompt, a frozen mid-game context from a real train-split rollout, and a probe turn asking whether
one statement is true evidence from the record. A fabrication is spliced back into the context that
produced it.

## Contents

One sidecar per policy size (`4B`, `8B`), 1,924 rows each: 481 of each of four kinds in two
length-matched blocks, 962 true / 962 false over the same games.

| block | kind | label | the statement | paper's name |
|---|---|---|---|---|
| A | `fabrication` | false | a judge-confirmed fabricated claim the policy made, in its own pre-fabrication context | Extracted Fabrication |
| A | `claim_real` | true | the evidence `Description:` line that claim's context quoted | Reference |
| B | `paraphrase_real` | true | a meaning-preserving rewrite of an unused record line | Paraphrased Reference |
| B | `altered_real` | false | the same line with one checkable detail changed | Perturbed Reference |

Block B removes the topic cue block A leaves (fabrications skew toward appeals to outside authority
and belief statements), since an `altered_real` item can only be caught by checking the record. This
is the training set, drawn from the RL split; the §4.3 detection readout uses separate items over the
held-out test set, built in
[`evaluation/source_of_fabrication/`](../../../evaluation/source_of_fabrication/README.md).

## Inputs

Step 1 reads, and cannot generate, a train-split rollout and its claim-level audit for each policy
size and each of three origin checkpoints (base, and steps 50 and 100 of a strategies-prompt GRPO
run):

```bash
# the rollout: one job per (size, origin); co-serves the sender and the 35B on 3 GPUs
export SENDER=<sender key> PROFILES=stubborn SPLIT=train END_IDX=921 PROMPT=strategies AUDIT_FAB=0
export IDS_FILE=$PWD/datasets/old_bailey/_generated/rl_sftsplit_strategies/sft_split.json
export IDS_KEY=rl_train_ids
sbatch --export=ALL scripts/oldbailey_rl_eval.slurm

# the claim-level audit (judge = the served 35B)
RECEIVER_HOST=<host> RECEIVER_PORT=<port> RECEIVER_MODEL_ID=qwen3.5-35B \
  python evaluation/source_of_fabrication/audit_rollouts.py \
      --result-file <rollout>.json \
      --full-template datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json \
      --out <report-dir>/audits/audit__<size>__<origin>__train.jsonl
```

A stock model played under the guide prompt also needs
`RES_TAG=_sender-$(python -m rl.sender_prompts --print result-stem --domain old-bailey --prompt strategies)`,
or it takes the file name of the base-prompt rollout of the same key. Step 1 finds the audits under
`REPORT_DIR` (`audits/audit__{size}__{ck}__train.jsonl`) and the rollouts through a built-in map under
`experiments/results/old-bailey`; `RESULTS_JSON`, a JSON object `{"<size>_<origin>": "<rollout path>"}`,
names them explicitly.

## The chain

```bash
REPORT_DIR=<report-dir> bash datasets/old_bailey/aux_loss/build_chain.sh   # every step
STEPS="4 5 6" bash datasets/old_bailey/aux_loss/build_chain.sh             # resume at the validation reference
DRY_RUN=1 bash datasets/old_bailey/aux_loss/build_chain.sh                 # print every command, run nothing
```

Knobs: `SIZES` (`"4B 8B"`), `STEPS` (`"1 2 3 4 5 6"`), `GEN` (the generated root), `REPORT_DIR`
(required whenever step 1 runs), `RESULTS_JSON`, `TOK_4B` / `TOK_8B` (the tokenizers the contexts are
checked with, default `$MODELS_DIR/Qwen3-4B-Instruct-2507` / `$MODELS_DIR/Qwen3-8B`), `PY`, `DRY_RUN`
(0). Only step 3 calls a model: it needs `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for the gateway
serving `--model`, default `gpt-5.4-mini`) and network access to that endpoint, and its on-disk cache
makes a repeat build free and byte-identical. Each stage writes one directory under
`_generated/aux_ce_probe/variants/`:

| step | script | writes |
|---|---|---|
| 1 | `build_aux_ce_sidecar.py` | one in-role probe item per train game: a judge-confirmed fabrication in its own context (cap 3 per game per origin), and real record lines until true ≈ false |
| 2 | `filter_aux_ce_sidecar.py` | `base_origin_all` (false items from the base checkpoint only, true re-matched 1:1 from the same games), then `base_origin` (647 false items per size, nested inside it) |
| 3 | `rebalance_aux_ce_sidecar.py`, `check_statement_balance.py` | `balanced`: each true record line rewritten into a terse claim matched to a false partner's length and style, plus block B; the balance gate must pass |
| 4 | `build_auxce_sft_dataset.py` | the frozen 30-game validation reference later stages are checked against |
| 5 | `build_auxce_sft_dataset.py`, `make_train_sidecar_meta.py` | `kind_balanced`: 481 × 4 kinds = 1,924 rows at both sizes, and its `_meta.json` |
| 6 | `verbatim_claim_sidecar.py`, `check_statement_balance.py --expect-fail` | `final`: block A's true statement set back to the verbatim record line; every other row byte-identical to its parent |

The balance gate (`check_statement_balance.py`) fails a sidecar that is separable by surface form: a
length AUC outside `[0.45, 0.55]`, a best single length threshold above 0.55, an 8-feature surface-only
classifier above 0.55 under game-grouped cross-validation, a pair word-count gap above 3, or skewed
verbatim-overlap / all-caps rates. Every threshold is a flag. `final` fails it by construction (the
verbatim line is the paper's *Reference* row), so step 6 runs it with `--expect-fail`, which records
the confound as a number. `--sidecar <path> --json-out <file>` gates one file.

`make_train_sidecar_meta.py` checks each stage against its parent (sha256, row equality and order,
counts, whole pairs, the val split, token caps) and copies the parent's `chat_template_sha` rather
than recomputing it. To re-validate an existing stage without rebuilding:

```bash
python datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py    --tag base_origin_all --selfcheck-only
python datasets/old_bailey/aux_loss/rebalance_aux_ce_sidecar.py --tag balanced         --selfcheck-only
python datasets/old_bailey/aux_loss/verbatim_claim_sidecar.py                          --selfcheck-only
```

## Output and training

```
datasets/old_bailey/_generated/aux_ce_probe/variants/final/{4B,8B}/sidecar.jsonl.gz
datasets/old_bailey/_generated/aux_ce_probe/variants/final/_meta.json
```

The sidecar is gzipped JSON lines, one probe item per line. `_meta.json` records per size the
`sidecar_sha256`, the item / label / per-kind counts, `max_prompt_tokens` and `chat_template_sha`.
Training refuses to start when the token cap exceeds `AUX_CE_MAX_PROMPT`, the `chat_template_sha`
differs from the policy tokenizer's, the labels are not 1:1, or (under `AUX_CE_KIND_BALANCE=1`) a kind
or a block's label balance is missing.

```bash
sbatch --export=ALL,ALGO=base+aux_loss,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,ROLLOUT_SEED=2026,POLICY_MODEL=qwen3-4B \
    scripts/rl_train_sender.slurm
```

`AUX_CE_DATA` defaults to the path above with `{size}` from `POLICY_MODEL`. The other defaults
(`AUX_CE_COEFF=0.5`, `AUX_CE_SAMPLING=random`, `AUX_CE_KIND_BALANCE=1`, `AUX_CE_BALANCE_MODE=norm_ratio`,
`AUX_CE_TARGET_GRAD_RATIO=0.25`, `AUX_CE_MAX_PROMPT=8192`, `AUX_CE_ROW_FRAC=1.0`, `AUX_CE_SEED=2026`)
are printed by `python -m rl.arm_config --describe --algo base+aux_loss --prompt strategies --repo $PWD`
and each can be overridden. `aux_loss` requires `PROMPT=strategies` and conflicts with
`REJECTION_SAMPLING=1` ([`rl/README.md`](../../../rl/README.md#algorithm-components)). The derived
`EXP_NAME` does not encode `AUX_CE_DATA`, so a sidecar ablation needs an explicit `EXP_NAME`.

Steps 1–2 and 4–6 are deterministic given their inputs, but step 1's inputs are sampled rollouts and
judge labels, so a rebuild's item pool differs from the paper's. Step 3 is byte-identical against a
warm cache; `sidecar_sha256` records which build you have.
