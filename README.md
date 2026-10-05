<div align="center">

<h1 align="center" style="text-align:center; font-weight:bold; font-size:2.0em; letter-spacing:2.0px;">Understanding and Reducing Misalignment in Persuasion Optimization</h1>

<p align="center" style="text-align:center; font-size:1.25em;">
<a href="https://www.boyiwei.com/" target="_blank" style="text-decoration: none;">Boyi Wei<sup>1,2</sup></a>&nbsp;,&nbsp;
<a href="https://rutadesai.github.io/" target="_blank" style="text-decoration: none;">Ruta Desai<sup>1</sup></a>&nbsp;,&nbsp;
<a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Peter Henderson<sup>2</sup></a>&nbsp;,&nbsp;
<a href="https://wenyueh.github.io/en/" target="_blank" style="text-decoration: none;">Wenyue Hua<sup>1</sup></a>
<br>
<sup>1</sup>Microsoft Research&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Princeton University
</p>

</div>

## Repo Overview

```
.
├── agents/                 conversation loop
│   ├── agent_base.py
│   ├── agent_quality.py
│   ├── rollout.py
│   └── model/
├── config/
│   ├── available_models.yaml
│   ├── sender/            sender profiles
│   ├── receiver/
│   │   ├── persona/       receiver profiles
│   │   └── model/         receiver hosting config
│   └── experiment/
├── datasets/              persuasion datasets
│   ├── old_bailey/
│   ├── house_showing/
│   └── nutrition/
├── rl/                     RL training scripts
│   ├── config/
│   ├── arm_config.py
│   ├── sender_prompts.py
│   ├── reward_function.py
│   ├── persuasion_interaction.py
│   └── strategy_audit/
├── sft/
│   └── sft_dataset.py
├── evaluation/
│   ├── rl_rollout.py
│   ├── audit_fabrications.py
│   ├── old_bailey/
│   ├── house_showing/
│   ├── nutrition/
│   ├── reward_landscape/
│   └── source_of_fabrication/
└── scripts/                SLURM scripts
```

Each of `rl/`, `datasets/*/`, `datasets/old_bailey/{sft,aux_loss}/` and
`evaluation/{reward_landscape,source_of_fabrication}/` has its own README with the details.

## Environment Setup

```bash
export MODELS_DIR=/path/to/models                    # model checkpoints (default ./models)
export CACHE_DIR=/path/to/cache                      # HF, uv, wandb and JIT-kernel caches (default ./.cache)
export VERL_DIR=/path/to/verl                        # a verl 0.7.0.dev checkout (required, no default)
export UV_CACHE_DIR=${CACHE_DIR:-$PWD/.cache}/uv     # uv's package cache
uv sync --frozen                                     # creates ./.venv from uv.lock
bash scripts/rl_build_env.sh                         # patch verl and sglang, install verl into ./.venv (needs network)
bash scripts/make_private_verl.sh /abs/path/verl-<study>   # optional private patched verl
source .venv/bin/activate                            # the `python` commands below assume this venv
```

Every launcher reads the stock model checkpoints from `MODELS_DIR` (default `<repo>/models`), one
directory per model: `Qwen3-4B-Instruct-2507`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3.5-9B` and
`Qwen3.5-35B-A3B`. Caches go under `CACHE_DIR` (default `<repo>/.cache`), and a per-cache variable such
as `HF_HOME` or `XDG_CACHE_HOME` still overrides its own location. `rl_build_env.sh` requires
`VERL_DIR`, the verl checkout it patches, and checks the stack against `POLICY_PATH` (default
`$MODELS_DIR/Qwen3.5-9B`) ([details](rl/README.md#environment)).

## Datasets

| path | content |
|---|---|
| `datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json` | 1,225 annotated Old Bailey cases (1,221 with evidence) |
| `datasets/house_showing/processed/{full,clean}/house_showing_*.json` | 211 games: 5 features, each absent / present-true / present-false |
| `datasets/nutrition/processed/{full,clean}/nutrition_*.json` | 28 games: a fixed 5-claim truth set, one game per inclusion subset |

`full/` carries the annotations and `params.private.features`, the evaluator's ground truth; `clean/`
is the Description-only view the sender sees. See [Old Bailey](datasets/old_bailey/README.md),
[House Showing](datasets/house_showing/README.md) and [Nutrition](datasets/nutrition/README.md).

## Pipeline

### Data for training

`python datasets/old_bailey/split_rl_sft.py` holds out the 100 validation games (`random.Random(42)`)
that every evaluation and both packaged experiments play, and writes GRPO parquets per receiver
distribution (`bayesian`, `stubborn`) under `datasets/old_bailey/_generated/<root>/<dist>/`. With
`--sft-holdout 200 --sft-seed 2026` it also carves 200 SFT games, leaving 921 RL / 200 SFT / 100
validation. `scripts/rl_train_sender.slurm` builds the root it needs automatically
([details](datasets/old_bailey/README.md)).

The SFT inits are trained on audited demonstrations over the 200 SFT games: collect with
`datasets/old_bailey/sft/collect_rollouts.sh` (hosted teacher) or `collect_selfdistill.slurm`
(self-distillation), build parquets with `build_sft_dataset.py`, and train with
`scripts/sft_train_sender.slurm`, whose `hf_final` directory GRPO takes as `SFT_INIT`
([details](datasets/old_bailey/sft/README.md)).

`ALGO=...+aux_loss` trains against a probe sidecar of in-role true/false statements, built from
train-split rollouts and their claim-level audits by
`REPORT_DIR=<claim-level audits> bash datasets/old_bailey/aux_loss/build_chain.sh`. Step 3 calls a
hosted model and needs `OPENAI_API_KEY` and network access. The output,
`datasets/old_bailey/_generated/aux_ce_probe/variants/final/{4B,8B}/sidecar.jsonl.gz`, is the
`AUX_CE_DATA` default ([details](datasets/old_bailey/aux_loss/README.md)).

### GRPO training

`scripts/rl_train_sender.slurm` has two axes. `ALGO` is what is optimized: `base` (the juror's final
belief minus its prior, plus a format bonus), `+penalty` (minus a coefficient times the 11
misaligned-technique monitors) and `+aux_loss` (an auxiliary cross-entropy awareness loss on the
probe sidecar). `PROMPT` is what the sender is told:

| `PROMPT` | Name in Paper | run-name suffix |
|---|---|---|
| `base` | *Base Prompt* | none |
| `strategies` | *Full Guidance Prompt* | `_strategies` |
| `single_strategy:<slug>` | *Single-Strategy Guidance Prompt* | `_only_<slug>` |

```bash
sbatch --export=ALL,ALGO=base,PROMPT=base,DIST=stubborn                                   scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+penalty,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1         scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+aux_loss,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,ROLLOUT_SEED=2026 scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base,PROMPT=single_strategy:framing,DIST=stubborn                scripts/rl_train_sender.slurm
# print the components, every variable they set and where each value came from:
python -m rl.arm_config --describe --algo base+penalty+aux_loss --prompt strategies \
    --policy-model qwen3-8B --repo $PWD
```

`DIST` picks the juror and the reward baseline: `bayesian` is the neutral juror at prior 0.5,
`stubborn` is prior 0.1. `POLICY_MODEL` is `qwen3-4B|qwen3-8B|qwen3-14B`; 4B and 8B run as one 6-GPU
job, and 14B needs a separate judge job. After training, `sbatch scripts/merge_verl_ckpt.slurm` merges
the checkpoint shards into the HuggingFace directory sglang serves. [`rl/README.md`](rl/README.md) has
every knob, the reward, run names, outputs and the 14B launch.

### Evaluation

One 3-GPU job per sender co-serves the sender (TP=1) and the receiver/judge (TP=2), plays every game
with `evaluation/rl_rollout.py` (which replays the verl GRPO conversation exactly), and then scores.
The receiver is the juror in Old Bailey, the buyer in house showing and the patient in nutrition.

```bash
sbatch --export=ALL,SENDER=<key> scripts/oldbailey_rl_eval.slurm
sbatch --export=ALL,SENDER=<key> scripts/houseshowing_rl_eval.slurm
sbatch --export=ALL,SENDER=<key> scripts/nutrition_rl_eval.slurm
# evaluate a checkpoint under the prompt it trained on, and tag the result file accordingly:
T=_sender-$(python -m rl.sender_prompts --print result-stem --domain house-showing --prompt strategies)
sbatch --export=ALL,SENDER=<key>,PROMPT=strategies,RES_TAG=$T scripts/houseshowing_rl_eval.slurm
```

`SENDER` is a model key the launcher's `resolve_model` maps to a checkpoint; for any other checkpoint
also pass `SENDER_MODEL_PATH=<hf dir>`. Defaults: `END_IDX` 100 / 211 / 28, `NUM_STEPS=3`,
`PROFILES="neutral stubborn"`, `MAX_WORKERS=16`, `RECEIVER=qwen3.5-35B`, `PROMPT=base`. Old Bailey adds
`SPLIT` (`val`), `VAL_PARQUET`, `IDS_FILE` / `IDS_KEY`, `AUDIT_FAB` (1) and `STRATEGY_AUDIT` (0).
Without `RES_TAG`, a non-base-prompt run takes the base-prompt file name of that sender key. Results
land under `experiments/results/`:

```
old-bailey/<sender>/<profile>_oldbailey_rlrollout_<split><RES_TAG>__recv_<receiver>.json
house-showing/<sender>/<profile>_house_showing_rlrollout<RES_TAG>__recv_<receiver>.json
nutrition/<sender>/<profile>_nutrition_rlrollout<RES_TAG>__recv_<receiver>.json
```

Receiver models live in `config/receiver/model/` (`python evaluation/receiver_models.py --list`):

| id | transport | where it runs | result stem | role |
|---|---|---|---|---|
| `qwen3.5-35B` | `sglang` | a GPU job, co-served by the eval launchers or `scripts/serve_receiver_xnode.slurm` | `__recv_qwen3.5-35B` | the main receiver, and always the audit judge (default) |
| `DeepSeek-V4-Flash` | `gateway` | `scripts/hosted_juror_login_driver.sh`, outside SLURM | `__recv_DeepSeek-V4-Flash` | the second receiver of the transferability check (paper App. D.3) |

Training always uses the local Qwen3.5-35B. The SLURM launchers reject `RECEIVER=DeepSeek-V4-Flash`;
play the hosted receiver with `scripts/hosted_juror_login_driver.sh`, which runs outside SLURM on a
machine with API access and plays against a sender served by a separate job. Hosted models go through
an OpenAI-compatible API gateway: set `GATEWAY_URL` to its chat-completions endpoint and
`GATEWAY_API_KEY` to your key (neither has a default).

```bash
# cells.tsv: one row per sender to serve, "cell_key<TAB>slug<TAB>ckpt_path<TAB>stem_kind"
sbatch --export=ALL,CELLS_FILE=$PWD/cells.tsv,ENDPOINT_FILE=$PWD/logs/sender_ep.txt scripts/serve_sender_xnode.slurm
sbatch --export=ALL,RECEIVER=qwen3.5-35B,ENDPOINT_FILE=$PWD/logs/judge_ep.txt       scripts/serve_receiver_xnode.slurm
ENDPOINT_FILE=$PWD/logs/sender_ep.txt RECEIVER_ENDPOINT="$(cat logs/judge_ep.txt)" \
    STRATEGY_AUDIT=1 bash scripts/hosted_juror_login_driver.sh
```

`gpt-5.4-mini` and `gpt-5.5` as senders use the same gateway (`GATEWAY_URL`, `GATEWAY_API_KEY`) and a
machine with API access:

```bash
RECEIVER_HOST=${EP%%:*} RECEIVER_PORT=${EP##*:} RECEIVER_MODEL_ID=qwen3.5-35B \
  python evaluation/rl_rollout.py --sender-api gateway --sender-name gpt-5.5 \
      --receiver-api sglang --domain old-bailey --profile stubborn --split val \
      --val-parquet datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet \
      --max-workers 8 --out experiments/results/old-bailey/gpt-5.5/stubborn_oldbailey_rlrollout_val__recv_qwen3.5-35B.json
```

### Strategy audits

Both audits use the served Qwen3.5-35B as judge, whichever receiver played the games. Run the
fabrication audit first, because the strategy audit's `false_information` column reads its sidecar
(otherwise Old Bailey takes a second judge draw and the OOD domains error out).

```bash
python evaluation/audit_fabrications.py --domain old-bailey --result-file <res> \
    --full-template datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json \
    --model <sender> --receiver-config <profile> --max-workers 16 --no-wandb
python evaluation/house_showing/evaluate_houseshowing.py --result-file <res> \
    --full-template datasets/house_showing/processed/full/house_showing_full.json \
    --model <sender> --receiver-config <profile> --max-workers 16 --no-wandb
python -m rl.strategy_audit.run --result <res> --domain house-showing --max-workers 16
```

`evaluation/audit_fabrications.py` covers all three domains; `evaluation/house_showing/evaluate_houseshowing.py`
and `evaluation/nutrition/evaluate_nutrition.py` are thin wrappers with the same CLI. One 4096-token
judge call per game counts the sender's `false_claims` against the exact record.
