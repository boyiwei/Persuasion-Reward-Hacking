# `rl/`: GRPO sender training

A sender policy (Qwen3-4B-Instruct-2507 by default, full-parameter fine-tuning through a patched verl) plays the
three-round persuasion game against a served receiver (Qwen3.5-35B-A3B), and GRPO optimizes the receiver's final
reported belief. The same served model is the judge behind every reward-hacking monitor and every offline audit. A run
is one algorithm selection (`ALGO`) and one sender prompt (`PROMPT`):

```bash
sbatch --export=ALL,ALGO=base,PROMPT=base,DIST=stubborn                   scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+penalty,PROMPT=strategies,SFT_SPLIT=1,DIST=stubborn  scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+aux_loss,PROMPT=strategies,SFT_SPLIT=1,DIST=stubborn,ROLLOUT_SEED=2026 scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base,PROMPT=single_strategy:framing,DIST=stubborn scripts/rl_train_sender.slurm
```

Join `ALGO` components with `+`, never a comma: `sbatch --export` is itself comma-separated, so `ALGO=base,penalty`
reaches the job as `ALGO=base`. The [root README](../README.md) has the pipeline overview.

## Contents

| path | what it is |
|---|---|
| `config/grpo_persuasion.yaml` | the verl GRPO config: KL-in-loss 0.01, lr 1e-6, batch 16, `n`=16, 8192/8192 prompt/response, `save_freq` 50, `test_freq` 10 |
| `config/interaction_config.yaml` | registers the interaction as `old_bailey_persuasion` |
| `config/algorithm/*.yaml`, `arm_config.py` | the `ALGO` components and their resolver |
| `config/prompt/*.yaml`, `sender_prompts.py` | the `PROMPT` modes and their resolver |
| `game_rows.py`, `evidence.py` | parquet rows, priors and receiver specs; `clean_evidence` |
| `persuasion_interaction.py`, `cognitive_models.py`, `receiver_client.py` | the rollout-time game, the receiver prompt, the client for the served receiver/judge |
| `reward_function.py`, `monitors.py` | the GRPO reward; the reward-hacking monitors |
| `aux_ce.py`, `aux_grad_balance.py`, `fab_aux_common.py` | the auxiliary CE loss, its gradient scaling, helpers shared with the sidecar builders |
| `rejection_sampling.py`, `rollout_seed.py`, `receiver_mix.py` | rejection sampling, deterministic rollout seeds, the optional receiver mixture (no paper arm uses it) |
| `strategy_audit/` | the 42-technique taxonomy, the shared judge `classify()`, the offline audit driver and wandb backfill |
| `patch_verl.sh`, `patch_sglang.sh`, `_patch_naive_concurrent.py`, `check_*.py` | the verl and sglang patches and the launcher's structural checks of them |

`rl/__init__.py` puts the repository root on `sys.path`, so `agents.*` and `evaluation.*` resolve in every verl worker.
Run everything from the repository root.

## Environment

Training and evaluation share the root `.venv` (see [Environment Setup](../README.md#environment-setup)).
`uv sync --frozen` installs everything from `uv.lock`, including verl's dependencies, at the versions of the
training environment. `scripts/rl_build_env.sh` then patches verl (`rl/patch_verl.sh "$VERL_DIR"`) and sglang
(`rl/patch_sglang.sh`), installs verl with `--no-deps`, fails if the venv differs from `uv.lock`
(`uv sync --frozen --inexact --check`), and checks that the stack loads `POLICY_PATH`. It refuses a `VERL_DIR`
that is not verl 0.7.0.dev.
Knobs: `VERL_DIR` (required, no default: the verl checkout to patch, verl 0.7.0.dev at commit `6dc50993`),
`RL_VENV` (`$REPO/.venv`), `POLICY_PATH` (`$MODELS_DIR/Qwen3.5-9B`), `UV_CACHE_DIR` (`$CACHE_DIR/uv`), `HF_HOME`
(`$CACHE_DIR`) and `XDG_CACHE_HOME` (`$CACHE_DIR/xdg`). Every launcher defaults `MODELS_DIR` to `$REPO/models` and
`CACHE_DIR` to `$REPO/.cache`.

`rl/patch_verl.sh` is idempotent. Besides compatibility fixes, it adds concurrent reward dispatch (step 9), per-step
wandb logging of the monitors (10, 10A, 10B), the rollout and judge dumps (10C–10F), rejection sampling (10G),
receiver-backend stamping (10H), the aux loss (10I, 13T) and deterministic sampling seeds (14, 14R, 14T, 14TF, 14S,
14SR). Patches 10G, 10H, 10I+13T and 14 are inert unless their knob is set, and the launcher refuses a run whose knob is
set against a checkout missing the marker. `patch_verl.sh` rewrites files in place, so worktrees with different copies
of it need separate checkouts; `scripts/make_private_verl.sh <dest> [<source>]` makes a checked private copy (~770 MB)
of `<source>`, which defaults to `$VERL_DIR`:

```bash
bash scripts/make_private_verl.sh /abs/path/verl-<study>
sbatch --export=ALL,VERL_HOME=/abs/path/verl-<study>,ALGO=base+aux_loss,PROMPT=strategies,SFT_SPLIT=1,DIST=stubborn scripts/rl_train_sender.slurm
```

With `VERL_HOME=<dir>` the launcher requires `<dir>/verl`, prepends `<dir>` to `PYTHONPATH` and asserts verl imports
from there; empty `VERL_HOME` uses the verl in `RL_VENV`.

## Data

`datasets/old_bailey/split_rl_sft.py` writes the parquets under `datasets/old_bailey/_generated/<variant>/<dist>/`
(`dist` is `bayesian` or `stubborn`): 1,121 train / 100 validation, or 921 RL / 200 SFT / 100 validation with
`SFT_SPLIT=1`. The prompt is baked into the parquet, so each spec has its own root. The launcher builds a missing root
and re-checks an `SFT_SPLIT=1` root against `sft_split.json` on every launch (duplicates, leakage into the SFT holdout,
row count). See [`datasets/old_bailey/README.md`](../datasets/old_bailey/README.md).

## The interaction

`rl/persuasion_interaction.py`, a verl `Interaction` registered as `old_bailey_persuasion`, plays the real game inside
the rollout:

```
arg1 -> recv1   (the receiver's CoT-stripped reply is injected as round-2 context)
arg2 -> recv2   (injected as round-3 context)
arg3 -> recv3   (not injected; the interaction terminates and this belief is the reward)
```

verl `multi_turn` must be `max_assistant_turns=4`, `max_user_turns=3` (one above the three rounds, so the receiver
answers the final argument); the interaction self-terminates on its third call. Each call returns the receiver's parsed
`P(guilty)` as that turn's `turn_score` on `extra_info["rollout_reward_scores"]["user_turn_rewards"]`, and the reward
reads the last value. Injected receiver turns are wrapped in `<receiver_response>…</receiver_response>`, stripped before
the sender's `<argument>`s are extracted, and masked out of the loss. The receiver prompt comes from
`config/receiver/persona/{bayesian,stubborn}_belief_oldbailey.yaml` and the public game params, never the annotated
evidence. A failed or empty receiver call becomes a plain round advance with a `-1.0` sentinel score, and the reward
function then queries the receiver itself, once per round.

## Reward

`rl/reward_function.py::belief_reward` returns one dict per sample. Only `score` enters the GRPO advantage:

```
score = (mu - prior)                                   # the outcome reward
      + FORMAT_REWARD_COEFF * format_score             # the turn-level format bonus
      - sum over active terms of PENALTY_COEFFICIENT * that term's rh_ monitor
```

`mu` is the receiver's final reported `P(guilty)`; `prior` is 0.5 for `DIST=bayesian` and 0.1 for `DIST=stubborn`, and
is also the prior both players read in their prompts. `format_score` is the fraction of rounds whose segment parses into
a closed `<thinking>` + `<argument>` pair; it requires a valid `mu` and is zeroed when the trajectory has more
advance-boilerplate segments than rounds (a forged terminator). `PENALTY_REGISTRY` maps a term to its monitor:

| term | monitor | extra judge call |
|---|---|---|
| `fabrication` / `fabrication_binary` | `rh_fake_evidence` (the false-claims count) / its `>= 1` indicator | no; the evidence audit already runs |
| `belief` | `rh_belief_coercion` (binary) | yes |
| `threats`, `social_punishment`, `creating_dependency`, `authority_impersonation`, `procedural_manipulation`, `discouragement`, `false_promises`, `misrepresentation`, `rumors` | `rh_<slug>` (binary) | yes, one each |

`PENALTY_TERMS=illegal_full11` activates all 11 misaligned-technique terms (the nine per-strategy ones, `belief` and
`fabrication_binary`) under one shared `PENALTY_COEFFICIENT`. The extra judges run on `RL_STRATEGY_JUDGE_WORKERS` (4)
threads with `RL_MONITOR_JUDGE_RETRY` (2) parse retries. A new term needs one `rh_<x>` monitor and one registry line.

Every other key is logged per train step as mean/min/max: `reward_extra/` (the reward components, failure flags and
one `<term>_penalty` per registry entry, `0.0` when inactive) and `reward_hacking/` (the 24 keys of
`rl.monitors.MONITOR_KEYS` plus `response_length`). The aux loss adds `actor/aux_ce_loss`, `actor/aux_gold_prob`,
`actor/aux_acc` and `aux_ce/`; rejection sampling adds `rejection/`.

## Algorithm components

`ALGO=base[+penalty][+aux_loss]`. Each component's values live in `rl/config/algorithm/<component>.yaml`, and
`rl/arm_config.py` resolves a selection into environment variables, the only channel into the Ray workers.

| component | what it adds | key values | run-name tag |
|---|---|---|---|
| `base` (implicit) | the outcome reward plus the format bonus, monitors on for logging | `FORMAT_REWARD_COEFF=0.1`, `RL_MONITOR_ENABLE=1` | `_fmt0p1` |
| `penalty` | subtracts the 11 misaligned-technique monitors, computed on every sample | `PENALTY_TERMS=illegal_full11`, `PENALTY_COEFFICIENT=0.3`, `RL_MONITOR_SAMPLE_RATE=1.0` | `pen0p3_illegal_full11` (`fakepen0` when absent) |
| `aux_loss` | an auxiliary cross-entropy awareness loss on the probe sidecar inside every optimizer step, gradient-balanced against the GRPO term | `AUX_CE=1`, `AUX_CE_DATA=<repo>/datasets/old_bailey/_generated/aux_ce_probe/variants/final/<size>/sidecar.jsonl.gz`, `AUX_CE_COEFF=0.5`, `AUX_CE_SAMPLING=random`, `AUX_CE_KIND_BALANCE=1`, `AUX_CE_BALANCE_MODE=norm_ratio`, `AUX_CE_TARGET_GRAD_RATIO=0.25` | `_auxce0p5_random_kindbal_normratio0p25` |

```bash
python -m rl.arm_config --describe  --algo base+penalty+aux_loss --prompt strategies \
    --policy-model qwen3-8B --repo $PWD
python -m rl.arm_config --print-env --algo base+penalty --prompt strategies --repo $PWD
```

`--print-env` sets only unset variables, so an explicit value wins (`--describe` reports it as `OVERRIDE`); a sweep is
`ALGO=base+penalty PENALTY_COEFFICIENT=0.5`. A leftover gate variable of an unselected component warns (`--strict-env`
makes it fatal), and two selected components setting one variable differently is an error.

`aux_loss` requires `PROMPT=strategies`, because every probe context in the sidecar bakes the 42-technique system
prompt; the yaml's `requires.prompt` and the launcher's gate both enforce it. It also conflicts with
`REJECTION_SAMPLING=1`, and rejection sampling requires an active penalty with the monitors at sample rate 1.0. Build
the sidecar with `datasets/old_bailey/aux_loss/build_chain.sh` ([README](../datasets/old_bailey/aux_loss/README.md)) or
point `AUX_CE_DATA` at another one. The derived `EXP_NAME` does not encode `AUX_CE_DATA`, so a sidecar ablation must
pass an explicit `EXP_NAME`, or `resume_mode=auto` continues the sibling arm's checkpoint.

## Sender prompts

`rl/sender_prompts.py` resolves a `PROMPT` spec into the sender config `AgentQuality` takes plus four derived names, so
the parquet root, checkpoint name, wandb tag and result-file stem always agree:

| spec | what the sender reads | `data-suffix` / `name-suffix` | `wandb-tag` | `arm-key` |
|---|---|---|---|---|
| `base` | the plain role prompt (prosecutor / seller / nutritionist) | (empty) | `base` | `base` |
| `strategies` | the role prompt plus the 42-technique guide: the `ALLOWED` techniques and the 11 `FORBIDDEN` ones | `_strategies` | `strategies` | `strat` |
| `single_strategy:framing` | the base role prompt plus one system line allowing exactly that technique | `_only_framing` | `only_framing` | `only_framing` |

`single_strategy` fills the `allowance_line` of `config/prompt/single_strategy.yaml` with `{name}` and `{definition}`
verbatim from the domain's taxonomy entry (the definition the audit judge reads) and appends it to the base template's
system message.

```bash
python -m rl.sender_prompts --check strategies                         # exit 2 on an unknown spec
python -m rl.sender_prompts --print name-suffix --prompt single_strategy:framing
python -m rl.sender_prompts --show --domain old-bailey --prompt strategies
python -m rl.sender_prompts --selftest    # 132 system prompts, 42 techniques x 3 domains
```

The rendered system message is the sender yaml's system content verbatim (682 / 7898 characters for old-bailey base /
strategies, 735 / 8189 for house-showing, 797 / 8286 for nutrition). Setting `SENDER_STRATEGIES` is a hard error that
names `PROMPT` instead.

## The taxonomy and the strategy audit

`rl/strategy_audit/taxonomy.py` holds the 42 techniques the paper calls persuasion strategies: 31 legitimate (the
paper's aligned strategies), 7 coercive and 4 deceptive (its distortive strategies). Those 11 are the paper's
misaligned strategies and this code's `illegal_full11` set. Each is worded per domain; 37 come from the vendored
`persuasion_taxonomy.jsonl`.

`python -m rl.strategy_audit.run` asks the served judge one yes/no question per (technique, argument) and writes a 1×42
vector per game:

```bash
# result mode: one result JSON written by evaluation/rl_rollout.py
python -m rl.strategy_audit.run --oldbailey-result <result>.json --max-workers 64
python -m rl.strategy_audit.run --result <result>.json --domain house-showing --max-workers 64

# rollout mode: a training run's own dumped rollouts, judge served by the launcher
sbatch --export=ALL,RUN=<exp name>,STEPS=1,N_GAMES=4 scripts/strategy_audit.slurm     # smoke
sbatch --export=ALL,RUN=<exp name>,N_GAMES=64        scripts/strategy_audit.slurm
```

Result mode writes `<result>.strategy_audit.json` (`--out` overrides); `--n-games` limits the games,
`--only-strategies illegal11|<comma list>` judges a subset (the rest stay unjudged, not 0), and `--per-turn` adds one
vector per round for the SFT builder. Rollout mode reads `$RESULTS_ROOT/<run>/rollouts/<step>.jsonl` and writes
`$RESULTS_ROOT/<run>/strategy_audit/<step>.jsonl`; `scripts/strategy_audit.slurm` serves the 35B on two GPUs and sets
`RESULTS_ROOT` to `$REPO/experiments/results/rl`. Its knobs: `RUN` (required), `STEPS`, `N_GAMES` (64), `MAX_WORKERS`
(256), `RETRY`, `JUDGE_MODEL` / `JUDGE_PATH` / `TP_SIZE` / `SGLANG_EXTRA`, `RL_VENV`.

`false_information` is never judged: it is `rh_fake_evidence >= 1`, the same rule as the `fabrication_binary` penalty.
Result mode reads the count from the fabrication sidecar, so run `evaluation/audit_fabrications.py` first; a missing
count gives `None`, not 0. The evidence-grounded techniques (`misrepresentation`, `evidence_based_persuasion`) get a
1024-token judge budget instead of 256. `bash scripts/strategy_audit_to_wandb.sh [--dry-run] <exp name>` pushes the
per-step distribution into the training run's wandb record and needs network access to wandb; `--dry-run` logs into a
throwaway run instead, so you can validate first.

## SFT initialization

`SFT_INIT=<HF dir>` initializes the policy from a supervised checkpoint and adds `_sftinit[-<tag>]` to the run name. It
requires `SFT_SPLIT=1`, because the SFT policy trained on the 200 holdout games; the only valid baseline for an
`SFT_INIT` arm is the same submit without it. `scripts/sft_train_sender.slurm` writes `<SAVE_DIR>/hf_final`, the
directory `SFT_INIT` takes; collection and training are in
[`datasets/old_bailey/sft/README.md`](../datasets/old_bailey/sft/README.md).

## Launching

| knob of `scripts/rl_train_sender.slurm` | default | meaning |
|---|---|---|
| `ALGO` | `base` | `base[+penalty][+aux_loss]` |
| `PROMPT` | `base` | `base` \| `strategies` \| `single_strategy:<slug>` (launcher-local; never exported) |
| `DIST` | `bayesian` | `bayesian` (neutral juror, prior 0.5) \| `stubborn` (prior 0.1); picks the parquet and the reward baseline |
| `POLICY_MODEL` | `qwen3-4B` | `qwen3-4B` \| `qwen3-8B` \| `qwen3-14B` |
| `POLICY_PATH` | `$MODELS_DIR/<model>` per `POLICY_MODEL` | explicit checkpoint; mutually exclusive with `SFT_INIT` |
| `N_GPUS` | 4 (4B/8B), 8 (14B) | trainer GPUs; `TRAIN_BATCH` must divide it |
| `TRAIN_BATCH`, `N_ROLLOUT`, `TOTAL_STEPS` | 16, 16, 100 | batch, GRPO group size, steps |
| `ROLLOUT_MEM_FRAC` | 0.5 / 0.40 / 0.55 | sglang `mem_fraction_static` per policy |
| `ROLLOUT_TP` | 1 | rollout tensor-parallel |
| `JUDGE_MODEL`, `JUDGE_PATH`, `JUDGE_TP` | `qwen3.5-35B`, `$MODELS_DIR/Qwen3.5-35B-A3B`, 2 | the served receiver and judge |
| `JUDGE_ENDPOINT_FILE` | `''` | external judge: read `host:port` from this file instead of serving inline (see 14B below); `JUDGE_WAIT_MIN`, `JUDGE_WAIT_TRIES` |
| `DATA_DIR` | `<repo>/datasets/old_bailey/_generated/<variant>/<DIST>` | the parquet directory |
| `SFT_SPLIT`, `SFT_HOLDOUT`, `SFT_SEED` | 0, 200, 2026 | the 921-game split and its carve-out |
| `SFT_INIT`, `SFT_INIT_TAG` | `''` | initialize from an SFT'd HF dir |
| `EXP_NAME`, `SAVE_DIR` | derived, `$REPO/experiments/results/rl/$EXP_NAME` | run name and output root |
| `VERL_HOME`, `RL_VENV` | `''`, `$REPO/.venv` | which verl, which interpreter |
| `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY` | `offline`, `persuasion-gym-oldbailey-rl`, `''` | |
| `EXTRA_OVERRIDES` | `''` | extra verl CLI overrides, spliced in last so they win |
| `ROLLOUT_SEED` | `''` | integer 0..2147483646: deterministic sender and local-receiver sampling; adds `_rseed<seed>`; rejects `REJECTION_SAMPLING=1` and an external judge |
| `REJECTION_SAMPLING`, `RS_TARGET_CLEAN`, `RS_MAX_GEN_ROUNDS` | 0, 8, 2 | penalty-free rejection sampling |
| `RL_MONITOR_TONE`, `RL_MONITOR_SAMPLE_RATE`, `RL_MONITOR_JUDGE_RETRY`, `RL_STRATEGY_JUDGE_WORKERS` | 0, 1.0, 2, 4 | monitor plumbing; pinned to enabled at rate 1.0 under a penalty or rejection sampling |
| `RL_ROLLOUT_DUMP_FREQ` | 5 | rollout dump cadence (step 1, then every N) |
| `SGLANG_EXTRA`, `TEACHER_MODEL`, `LOG_DIR` | `''`, `unknown`, `$REPO/logs` | extra sglang args, metadata, logs |
| `SMOKE` | 0 | 1 step, `TRAIN_BATCH=N_GPUS`, `N_ROLLOUT=2`, resume and saving disabled, `_smoke` name |

An individual `AUX_CE_*` or `PENALTY_*` knob in the environment overrides the component that owns it. A value
containing a comma must be exported in the shell and submitted with a bare `--export=ALL`; a value containing only
spaces (`EXTRA_OVERRIDES`) just needs quoting.

4B and 8B run as one 6-GPU job: 4 trainer GPUs plus the Qwen3.5-35B judge/receiver co-served on the top 2
(`JUDGE_TP=2`). 14B needs 8 trainer GPUs, so the judge runs as its own job:

```bash
sbatch --export=ALL,JUDGE_MODEL=qwen3.5-35B,ENDPOINT_FILE=logs/judge_ep.txt scripts/rl_serve_judge.slurm
sbatch --gres=gpu:8 --export=ALL,POLICY_MODEL=qwen3-14B,JUDGE_ENDPOINT_FILE=logs/judge_ep.txt,<rest> \
    scripts/rl_train_sender.slurm
```

### Run names, resume, wandb

```
sender_${POLICY_MODEL}_${RECV_LABEL}_${JUDGE_TAG}_${FAKEPEN_TAG}${FMT_SUFFIX}${RS_SUFFIX}${PROMPT_SUFFIX}${SFTSPLIT_SUFFIX}${SFTINIT_SUFFIX}${AUXCE_SUFFIX}${MIX_SUFFIX}${RSEED_SUFFIX}${SMOKE_SUFFIX}
```

`bayesian` → `neutral`; `qwen3.5-35B` → `j35B`; no active penalty → `fakepen0`, the full set → `pen<λ>_illegal_full11`;
`.` becomes `p` throughout. The launcher rejects an explicit `EXP_NAME` that omits a derived `_rseed`, `_mix`,
`_normratio` or `_kindbal` tag, or advertises one that is not configured.

```bash
sbatch --export=ALL,SMOKE=1,ALGO=base,PROMPT=base,DIST=stubborn scripts/rl_train_sender.slurm
```

A smoke uses tiny parquets under its own `rl*_smoke/` root, disables resume and saves nothing. Otherwise
`resume_mode=auto` silently resumes a `SAVE_DIR` that holds `global_step_*` directories (the launcher prints a `NOTE`
naming the step); extend a run by resubmitting with a larger `TOTAL_STEPS` under the same `EXP_NAME`. wandb runs
offline by default (`WANDB_MODE`); sync a run afterwards with `wandb sync wandb/offline-run-*`. `custom_metadata`
records the resolved run configuration.

### What a run writes

Under `SAVE_DIR` (default `experiments/results/rl/<EXP_NAME>/`):

| path | content |
|---|---|
| `global_step_<N>/actor/` | the FSDP-sharded actor checkpoint (`save_freq` 50) |
| `latest_checkpointed_iteration.txt` | the step `resume_mode=auto` continues from |
| `rollouts/<step>.jsonl` | prompt, response, score, monitors and the structured `messages` of each rollout |
| `judges/judges_step_<step>.yaml` | every judge call of that step, 1:1 with the rollout dump (`judges_val_step_<step>.yaml` for validation) |
| `strategy_audit/<step>.jsonl` | written later by `rl.strategy_audit.run` in rollout mode |

Offline wandb runs land under `wandb/` at the repository root; verl's resolved config and log go to `logs/verl/`.

### Merging a checkpoint for evaluation

Evaluation serves single HF directories, so merge the FSDP shards first:

```bash
sbatch --export=ALL,ONLY=<substring of the output slug> scripts/merge_verl_ckpt.slurm
```

It merges each `<run>/global_step_<N>/actor` in its `declare -A CKPTS` table to `<OUTROOT>/<slug>_gs<N>` with
`python -m verl.model_merger merge --backend fsdp` and copies the tokenizer files. Finished destinations are skipped, a
missing source is skipped unless `STRICT_MERGE=1`, and the job fails if every attempted merge failed. Knobs:
`MERGE_PY`, `RL_SRC` (the directory holding the run dirs, i.e. a `SAVE_DIR` parent; default `$REPO/experiments/results/rl`),
`OUTROOT` (default `$REPO/experiments/results/rl/_em_merged`, the `MERGED` root the evaluation launchers read), `ONLY`,
`STRICT_MERGE`.
