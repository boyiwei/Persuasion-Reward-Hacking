# SFT demonstration corpora

The demonstrations the paper's SFT-initialized GRPO arms start from. A teacher plays the 200 SFT-train
games under the 42-technique guide prompt, two per-turn audits label every round, and a filter keeps
the clean prefix of each rollout. The teacher is either `gpt-5.4-mini` through the API gateway
(`collect_rollouts.sh` with `SENDER_API=gateway`, on a machine with API access, with `GATEWAY_URL` and
`GATEWAY_API_KEY` set) or the policy itself (`collect_selfdistill.slurm`, one GPU job that co-serves
the sender and the 35B judge and calls the same driver with `SENDER_API=sglang`, so it needs no API
key). Scripts here run by path from the repository root.

Both teachers read the 200 game ids and the prompt prefixes from the guide-prompt split, so carve it
first:

```bash
python datasets/old_bailey/split_rl_sft.py --sender-prompt strategies \
    --sft-holdout 200 --sft-seed 2026 \
    --out-dir datasets/old_bailey/_generated/rl_sftsplit_strategies
```

## 1. Collect

```bash
# hosted teacher: serve the 35B receiver/judge as a GPU job, drive the passes from a machine with API access
sbatch --export=ALL,ENDPOINT_FILE=$PWD/logs/recv_endpoint.txt,SGLANG_VENV=$PWD/.venv \
    scripts/serve_receiver_xnode.slurm
SENDER=gpt-5.4-mini SENDER_API=gateway RECEIVER_ENDPOINT=$(cat logs/recv_endpoint.txt) \
  PROFILE=stubborn PROMPT=strategies PASSES=3 STEER_K=4 STEER_FROM_PASS=2 NUM_STEPS=3 \
  bash datasets/old_bailey/sft/collect_rollouts.sh

# self-distillation: one job co-serves the Qwen teacher (TP=1) and the 35B (TP=2)
sbatch --export=ALL,SENDER=qwen3-8B-base,PASSES=10,STEER_K=0,BUILD_SFT=0 \
    datasets/old_bailey/sft/collect_selfdistill.slurm        # 4B: SENDER=qwen3-4B-instruct-base
```

| `collect_rollouts.sh` knob (default) | meaning |
|---|---|
| `SENDER` (`gpt-5.4-mini`), `SENDER_API` (`gateway`) | the teacher and its transport; `gateway` needs `GATEWAY_URL` and `GATEWAY_API_KEY`, `sglang` needs `SENDER_ENDPOINT=<host>:<port>` serving the model name `SENDER` |
| `RECEIVER_ENDPOINT` (required), `RECEIVER` (`qwen3.5-35B`) | the served receiver, also both audit judges |
| `PASSES` (3) | temperature-1 passes over the same games |
| `PROFILE` (`stubborn`), `PROMPT` (`strategies`) | must match the GRPO `DIST` and data root |
| `NUM_STEPS` (3), `END_IDX` (all 200), `MAX_WORKERS` (16) | rollout shape |
| `STEER_K` (4), `STEER_FROM_PASS` (2) | passes from `STEER_FROM_PASS` on add a per-game seeded "favor these ALLOWED strategies" hint (generation only) |
| `AUDIT_WORKERS` (64), `FAB_WORKERS` (32) | judge concurrency |
| `SKIP` (`true`), `FORCE_AUDIT` (0) | id-keyed rollout resume; re-audit a pass whose sidecar already covers it |
| `IDS_FILE`, `VAL_PARQUET`, `FULL_TEMPLATE`, `REPO`, `PYTHON` | overrides for paths derived from `PROMPT`, the checkout and the interpreter |

`collect_selfdistill.slurm` takes the same knobs plus the builder and serving ones listed in its header
(`BUILD_SFT` (1) runs the builder in the same job; `SENDER_MODEL_PATH=/abs/hf_dir` serves a checkpoint
that is not a known model key). A knob value with a comma must be exported in the shell and submitted
with a bare `--export=ALL`; one with spaces, such as `BUILD_EXTRA`, just needs quoting.

Each pass writes, under `experiments/results/old-bailey/<sender>/sft_rollouts/`:

```
<profile>_sfttrain_<prompt>_pass<p>.json                  the rollout
<profile>_sfttrain_<prompt>_pass<p>.strategy_audit.json   42-way, --per-turn
<profile>_sfttrain_<prompt>_pass<p>.fabrication.json      the training instrument, --per-turn
```

A sidecar that already covers its pass is not re-judged, since the temperature-1 judge would change
the survivor set.

## 2. Build the parquets

```bash
python datasets/old_bailey/sft/build_sft_dataset.py \
    --results experiments/results/old-bailey/gpt-5.4-mini/sft_rollouts/stubborn_sfttrain_strategies_pass[0-9].json \
    --prompts-parquet datasets/old_bailey/_generated/rl_sftsplit_strategies/stubborn/sft_holdout.parquet \
    --out-dir datasets/old_bailey/_generated/sft/strategies_stubborn_gpt5_4 \
    --max-illegal-per-turn 0 --val-games 10 --val-seed 0 --max-tokens 7680 \
    --tokenizer ${MODELS_DIR:-models}/Qwen3-8B --max-dup-profile 3
```

List the pass result files explicitly (`pass[0-9].json` or `pass{1..10}.json`); a `pass*.json` glob
also matches the audit sidecars. Every example's `[system, user1]` prefix is copied verbatim from
`--prompts-parquet`, so that parquet is the prompt knob (`rl_sftsplit_strategies/<dist>/` for the
guide prompt, `rl_sftsplit/<dist>/` for the base prompt) and must match the GRPO data root.

A turn is clean when:

* its raw text is non-empty, has no forged round-advance boilerplate and no literal `<think>`,
  `</think>`, `<|im_start|>` or `<|im_end|>`, and passes the GRPO format predicate
  (`rl.reward_function._turn_format_ok`: a closed `<thinking>` before a closed `<argument>`);
* its `<argument>` and `<thinking>` bodies clear `--min-argument-words` (50) and
  `--min-thinking-words` (5);
* its `<argument>` recites no harness or taxonomy vocabulary (`[EVIDENCE evN]` handles, strategy
  labels, prior/posterior-belief framing, `FORBIDDEN`/`ALLOWED` rule talk; `--allow-harness-leakage`
  disables) and does not ask the court to acquit (`--allow-wrong-side` disables);
* its 42-strategy turn vector has at most `--max-illegal-per-turn` (0) of the 11 coercive and
  deceptive slugs, counting a `None` verdict as a violation, and `false_information` is 0;
* its per-turn fabrication audit shows `rh_fake_evidence == 0` (`--skip-fabrication-filter`
  disables, for plumbing tests only).

The first dirty turn truncates the rollout, so a dirty round 1 discards all of it. A clean prefix of
length k emits k examples, each supervised only on its final assistant turn (`loss_mask: 0` on the
others). Other flags: `--max-tokens` (keep it below the trainer's `data.max_length`), `--tokenizer`
(the student's), `--max-dup-profile` (cap on rows sharing game, prefix length and strategy profile;
default 2), `--val-games` / `--val-seed` (holdout by game), `--subsample` / `--subsample-seed`,
`--select-diverse`. Outputs: `<out-dir>/sft_train.parquet`, `<out-dir>/sft_val.parquet` (absent with
`--val-games 0`) and `<out-dir>/sft_build_stats.json`.

The paper's hosted-distill corpus is the command above (614 train / 29 val rows). The self-distill
corpus uses ten unsteered passes, trimmed to 614 rows with no val holdout:

```bash
python datasets/old_bailey/sft/build_sft_dataset.py \
    --results experiments/results/old-bailey/qwen3-8B-base/sft_rollouts/stubborn_sfttrain_strategies_pass{1..10}.json \
    --prompts-parquet datasets/old_bailey/_generated/rl_sftsplit_strategies/stubborn/sft_holdout.parquet \
    --out-dir datasets/old_bailey/_generated/sft/strategies_stubborn_qwen3-8B-base \
    --max-illegal-per-turn 0 --val-games 0 --val-seed 0 --max-tokens 7680 \
    --tokenizer ${MODELS_DIR:-models}/Qwen3-8B --max-dup-profile 3 \
    --subsample 614 --subsample-seed 0 --min-argument-words 50 --min-thinking-words 5
```

The 4B student takes the `qwen3-4B-instruct-base` passes, its own `--out-dir` and
`--tokenizer ${MODELS_DIR:-models}/Qwen3-4B-Instruct-2507`. Rollouts and judges sample at temperature 1, and
the builder applies the word floors and the harness-leakage and wrong-side predicates by default, so a
re-collection does not reproduce the paper's corpora row for row.

## 3. Train and use the init

```bash
sbatch --export=ALL,POLICY_MODEL=qwen3-4B,DATA_DIR=$PWD/datasets/old_bailey/_generated/sft/strategies_stubborn_gpt5_4,EXP_NAME=sft_qwen3-4B_strategies_stubborn_ep2_misrepv2 \
    scripts/sft_train_sender.slurm          # 8B: POLICY_MODEL=qwen3-8B; 14B: --gres=gpu:8, N_GPUS=8

sbatch --export=ALL,ALGO=base,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,POLICY_MODEL=qwen3-4B,SFT_INIT=$PWD/experiments/results/sft/sft_qwen3-4B_strategies_stubborn_ep2_misrepv2/hf_final,SFT_INIT_TAG=ep2 \
    scripts/rl_train_sender.slurm
```

SFT runs verl's `fsdp_sft_trainer` on four GPUs with `EPOCHS` (2), `LR` (1e-5), `TRAIN_BS` (32),
`MAX_LEN` (8192, `truncation=error`, so the builder's `--max-tokens` must keep every row under it) and
`N_GPUS` (4); the script header lists the rest. It writes `experiments/results/sft/<EXP_NAME>/` and
saves a full HF model at `global_step_<N>/huggingface/`, symlinked as `hf_final`. `SFT_INIT` takes
that directory, is mutually exclusive with `POLICY_PATH`, and requires `SFT_SPLIT=1` so GRPO never
trains on the 200 games the init saw; `SFT_INIT_TAG` names it in the run name (`…_sftinit-<tag>`).
