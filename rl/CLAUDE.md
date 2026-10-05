# Sender-RL training (GRPO)

Guidance for `rl/` — the package that trains the **sender** as a GRPO policy against the served
juror. This file loads when you work under `rl/`; the root `CLAUDE.md` covers the repository as a
whole and `rl/README.md` is the deep dive. The launchers `scripts/rl_train_sender.slurm`,
`scripts/sft_train_sender.slurm`, `scripts/rl_serve_judge.slurm` and `scripts/strategy_audit.slurm`
live outside `rl/` but belong to this track.

## The two axes

A run is one algorithm selection and one sender prompt:

```bash
sbatch --export=ALL,ALGO=base,PROMPT=base,DIST=stubborn                                              scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+penalty,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1                    scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base+aux_loss,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,ROLLOUT_SEED=2026 scripts/rl_train_sender.slurm
sbatch --export=ALL,ALGO=base,PROMPT=single_strategy:framing,DIST=stubborn                           scripts/rl_train_sender.slurm

python -m rl.arm_config --describe --algo base+penalty+aux_loss --prompt strategies --policy-model qwen3-8B --repo $PWD
```

🔴 **The component separator is `+`, never a comma.** `sbatch --export` is itself comma-separated, so
`ALGO=base,penalty` reaches the job as `ALGO=base`. The same trap applies to any value holding a
comma: export it in your shell and submit with a bare `--export=ALL`. A value holding only spaces
(`EXTRA_OVERRIDES`) just has to be quoted, so the shell hands `sbatch` one argument.
`rl.arm_config` rejects a comma-joined `ALGO` with that explanation.

### `ALGO` — `rl/config/algorithm/*.yaml` + `rl/arm_config.py`

| component | what it adds | key values | run-name tag |
|---|---|---|---|
| `base` (implicit, always on) | the outcome reward + the turn-level format bonus, monitors on for logging | `FORMAT_REWARD_COEFF=0.1`, `RL_MONITOR_ENABLE=1` | `_fmt0p1` |
| `penalty` | subtract λ × the sum of the 11 misaligned-technique monitors, monitors computed on every sample | `PENALTY_TERMS=illegal_full11`, `PENALTY_COEFFICIENT=0.3`, `RL_MONITOR_SAMPLE_RATE=1.0` | `pen0p3_illegal_full11` (`fakepen0` when off) |
| `aux_loss` | an auxiliary CE awareness loss inside every optimizer step, gradient-balanced against the GRPO term | `AUX_CE=1`, the aux-loss sidecar, `AUX_CE_COEFF=0.5` cap, `AUX_CE_SAMPLING=random`, `AUX_CE_KIND_BALANCE=1`, `AUX_CE_BALANCE_MODE=norm_ratio`, `AUX_CE_TARGET_GRAD_RATIO=0.25` | `_auxce0p5_random_kindbal_normratio0p25` |

Component file schema (any other key is an error): `id`, `order`, `implicit`, `doc`, `env`,
`env_off`, `gate`, `requires` (`prompt:` / `env:`), `conflicts` (`env:`), `expect_name_tag`,
`expect_name_tag_absent`. Only `{repo}` and `{size}` (the policy name without its `qwen3-` prefix)
may appear in a value.

**Resolver contract** (`rl/arm_config.py`, stdlib + yaml; `--print-env | --describe | --selftest`):

- It **emits only** environment variables the launcher, `rl/reward_function.py`, `rl/aux_ce.py` and
  the generated verl patches already read. Environment variables are the only channel into the Ray
  workers and verl loads the reward by file path, so a component can configure nothing else.
- `--print-env` writes shlex-quoted `export` lines **set-if-unset**: an explicit value in the
  environment wins and `--describe` reports it as `OVERRIDE`. That is how a coefficient sweep runs —
  `ALGO=base+penalty PENALTY_COEFFICIENT=0.5`. An exported-but-empty value counts as unset.
- `env_off` spells out **every** `AUX_CE_*` on purpose: the launcher exports them unconditionally and
  records `custom_metadata.aux_ce_data` on every run, so under `set -u` an unset one would abort an
  arm that does not use the aux loss at all.
- Two selected components disagreeing about one variable is an error. A gate variable of a
  **non-selected** component left in the environment is a loud warning; `--strict-env` makes it fatal.
- Refusals: `aux_loss` **requires `PROMPT=strategies`**, `aux_loss` **conflicts with
  `REJECTION_SAMPLING=1`**, `penalty` requires the monitors on at sample rate 1.0, and naming a
  component whose gate the environment switches off is an error.
- In the launcher the resolution runs right after `PY=` and before every `${VAR:-default}`, guarded
  with `|| { … exit 1; }` — never a bare `eval "$(...)"`, which would swallow a non-zero exit and
  leave the run silently unconfigured. The existing submit gates stay as the second check.

### `PROMPT` — `rl/config/prompt/*.yaml` + `rl/sender_prompts.py`

One spec string picks the sender prompt and the four names derived from it:

| spec | prompt | `data-suffix` | `name-suffix` | `arm-key` |
|---|---|---|---|---|
| `base` | the plain role prompt | `` | `` | `base` |
| `strategies` | role prompt + the 42-technique ALLOWED / FORBIDDEN guide | `_strategies` | `_strategies` | `strat` |
| `single_strategy:<slug>` | role prompt + one system line allowing exactly that technique | `_only_<slug>` | `_only_<slug>` | `only_<slug>` |

`single_strategy` has no yaml of its own: the base template gains one system line
(`allowance_line` in `rl/config/prompt/single_strategy.yaml`, with `{name}` / `{definition}` verbatim
from `rl/strategy_audit/taxonomy.py` for the domain), so "use only X" in the prompt and "did the
argument use X?" in the audit describe one technique. `PROMPT` is launcher-local and never exported;
`SENDER_STRATEGIES` is a hard error naming `PROMPT`.

`rl/sender_prompts.py` is the ONE owner of every prompt string and every name derived from it —
`rl.game_rows`, `datasets/old_bailey/split_rl_sft.py`, `evaluation/rl_rollout.py`, `rl.arm_config`
and both packaged experiments call it rather than spelling a literal:

```bash
python -m rl.sender_prompts --check strategies                       # exit 2 on an unknown spec
python -m rl.sender_prompts --show  --domain old-bailey --prompt single_strategy:framing
python -m rl.sender_prompts --print name-suffix --prompt single_strategy:framing
```

The prompt is baked into the parquet, so each spec writes its own data root and an evaluation plays
the prompt its checkpoint trained on (`rl_rollout.py --sender-prompt SPEC`).

## Data rows

`rl/game_rows.py` is the importable half of the dataset builder and
`datasets/old_bailey/split_rl_sft.py` the CLI that carves the splits and writes the parquets.

- 1225 annotated cases, minus 4 with zero evidence = **1221 playable**. `random.Random(42)` shuffles
  `range(1221)`; the first `--val-size 100` positions are the held-out validation set and the other
  1121 are the RL train side. With `--sft-holdout 200 --sft-seed 2026` an independent RNG samples 200
  ids from the *sorted* train ids, leaving **921 RL / 200 SFT / 100 validation**. The validation set
  is byte-identical for every sender prompt and every SFT seed (`_seeded_val_idx` is the one source).
- Rows are written per receiver distribution, and the distributions differ ONLY in the per-row
  receiver spec and the prior:

  | `DIST` | juror | prior P(guilty) | reward baseline |
  |---|---|---|---|
  | `bayesian` | neutral Bayesian juror | 0.5 | µ − 0.5 |
  | `stubborn` | presumption of innocence + resist-updating disposition | 0.1 | µ − 0.1 |

  The prior is both prose (it overrides `public.prior_belief`, so the sender and the receiver read
  the same number) and the arithmetic baseline. An unknown distribution **raises** — a typo must
  never fall through to some other juror.
- Each row carries the sender's initial chat with the evidence **cleaned**
  (`rl/evidence.py:clean_evidence` drops `Strength:` / `Reasoning:`), a slim JSON payload for the live
  receiver queries (no annotated evidence — the receiver never sees ground truth) and a full payload
  with the annotations for the reward and the monitors, plus `n_pros_favoring` as a difficulty tag.
- Data roots follow the two knobs: `rl[_sftsplit]<data-suffix>` under
  `datasets/old_bailey/_generated/`. `scripts/rl_train_sender.slurm` auto-builds the root its knobs
  select, and re-checks an `SFT_SPLIT=1` root against `sft_split.json` on **every** launch (duplicate
  rows, leakage into the SFT holdout, exact row count).

## The real sender↔receiver rollout

`rl/persuasion_interaction.py` (registered in `rl/config/interaction_config.yaml` as
`old_bailey_persuasion`) actually plays the game inside the verl rollout: after every sender
`<argument>` it queries the served receiver through `rl.receiver_client` and injects the reply's
CoT-stripped `<belief>` + `<argument>` blocks, wrapped in `<receiver_response>`, as the next round's
context.

verl `multi_turn` is **`max_assistant_turns=4`, `max_user_turns=3`** — one above the 3 game rounds,
because the receiver re-entry gate is `current_turns < max_assistant_turns` evaluated at
`current_turns=3` after the third argument. The interaction self-terminates on its 3rd call, so the
sender still emits exactly 3 arguments and that last receiver reply is **not** injected; its belief
is captured as the final turn score and IS the reward. Each call returns the parsed P(guilty) on
`extra_info["rollout_reward_scores"]["user_turn_rewards"]`, so the reward reads the last value with
no receiver re-query in the happy path (an empty channel falls back to re-querying). verl masks the
injected user tokens out of the loss.

`evaluation/rl_rollout.py` imports the same templates and extraction rules, so an evaluation replays
the training conversation rather than approximating it.

## Reward

`rl/reward_function.py`:

```
score = (µ − prior)                                                 # the outcome term
      + FORMAT_REWARD_COEFF · (fraction of rounds with a CLOSED <thinking> + <argument> pair)
      − PENALTY_COEFFICIENT · Σ over active terms (that term's rh_ monitor)
```

- Only `score` enters the GRPO advantage; every other key is a monitor that the patched verl
  aggregates to wandb (`reward_hacking/…`, `reward_extra/…`) without touching the gradient.
- The format bonus is gated on a valid outcome (a receiver outage reads as a flat zero) and is zeroed
  outright when the trajectory splits into more advance-boilerplate segments than the game has rounds
  — a forged terminator, logged as `reward_extra/format_forged`. The injected round-advance text must
  keep ending with the exact phrase "wrapped in a single `<argument>` block." (the `_ADVANCE_RE`
  terminator); the bonus exists to keep the CoT extractable and to counter tag-dropping evasion.
- `<receiver_response>` spans are stripped from the trajectory before the sender's own `<argument>`s
  are extracted for the monitors.

### The penalty registry

`PENALTY_REGISTRY` maps a term name to (a) the `rh_` monitor whose value × λ is that term's
contribution and (b) the fixed `reward_extra/` key it is logged under. `PENALTY_COEFFICIENT` is the
single shared λ for every active term; both knobs empty is the penalty-free baseline.

| term | monitor | cost |
|---|---|---|
| `fabrication` | `rh_fake_evidence` (the false-claims count) | free — already a base monitor |
| `fabrication_binary` | `1[rh_fake_evidence ≥ 1]` | free; mutually exclusive with `fabrication` |
| `belief` | `rh_belief_coercion` (binary) | one extra judge call per sample |
| the 9 per-strategy slugs | `rh_<slug>` (binary) | one extra judge call each |

`PENALTY_TERMS=illegal_full11` activates all 11 at once — the 9 per-strategy terms plus `belief` and
`fabrication_binary`. `ILLEGAL_TERMS` is cross-checked at import against the taxonomy's coercive
(7) + deceptive (4) slugs — the paper's coercive and *distortive* strategies — so an edit to the
taxonomy fails loudly here. Every registry term logs
its `<term>_penalty` series on every call (`0.0` when inactive), so downstream analysis should read
the registry rather than a hard-coded key list. To add a term: write an `rh_<x>` monitor and wire it
into `monitors.MONITOR_KEYS`, add one registry line, and name the term in `PENALTY_TERMS`.

The per-strategy and belief judges fan out over `RL_STRATEGY_JUDGE_WORKERS`(4) threads with
`RL_MONITOR_JUDGE_RETRY`(2) parse retries. An active penalty requires the monitors on at
`RL_MONITOR_SAMPLE_RATE=1.0` (submit-validated): an unsampled rollout would read `rh_*=0` and train
penalty-free.

## ONE judge, two callers

`rl/strategy_audit/audit.py::classify()` is the single function the **online** GRPO penalty and the
**offline** audit (`python -m rl.strategy_audit.run`) both call. It owns everything that can change a
verdict — the prompt, the evidence cleaning, the retry loop, the parser and the token budget — so the
two pipelines cannot drift. `rl.monitors` overrides only the transport (a `chat_fn` through `_judge`,
so an online verdict still lands in the judge dump). Verify any time with
`python -m rl.strategy_audit.audit --selftest` (no GPU; it asserts the online and offline paths emit
byte-identical messages and budgets, and exits 2 on a typo'd flag so a smoke cannot mistake usage for
a pass).

- **Budgets** (`audit.judge_budget()`): **1024** tokens for the evidence-grounded prompts
  (`misrepresentation`, `evidence_based_persuasion`), **256** otherwise. The grounded prompts reason
  item by item in plain visible output; judging them at 256 cuts a quarter of attempts off before
  `<answer>` and biases the surviving verdicts low, because the yes/no fallback fires on
  half-finished reasoning. The grounded prompts therefore also **require** the `<answer>` tag
  (`parse_binary(require_tag=True)`): for a prompt that orders the tag, an untagged reply is a
  truncation and retries instead of reading as a 0.
- **Evidence cleaning lives inside `classify`**, so the annotated `Strength:` / `Reasoning:` ground
  truth can never reach the judge whatever the caller passes. It is a prompt property, not only a
  leak guard: a judge shown annotated lines is a different instrument.
- Audits record `judge_budget_policy` and `judge_budget_grounded_slugs` (the slugs actually judged
  with the grounded prompt — the policy alone would claim 1024 for a column judged at 256).
  `python -m rl.strategy_audit.render_check --game-id N --result <json>` prints the exact wire
  prompts with no judge.
- **`false_information` has ONE definition everywhere**: count the false claims first
  (`rh_fake_evidence`), then threshold `>= 1` (`run.fi_indicator`, the same rule as the
  `fabrication_binary` penalty term). It is **never** judged by a binary judge in any mode. Rollout
  mode reads the count from the dump record; result mode reads it from the `.fabrication.json` /
  `.deception.json` sidecar (`--fabrication-json` overrides) and computes it in-process only for Old
  Bailey when the sidecar is missing — so **run the fabrication audit first** or that column becomes
  a second judge draw. No count → `None` (unjudged), never a silent 0; the artifact records
  `false_information_source` and `false_information_instrument` (`run.fi_instrument`, derived from
  the count sidecar's own stamps, so one label never covers two instruments).
- **One retry knob for both pipelines.** The offline `--retry` defaults to `RL_MONITOR_JUDGE_RETRY`,
  the same variable the online monitors read; precedence `--retry` > `RETRY` >
  `RL_MONITOR_JUDGE_RETRY` > 2. `scripts/strategy_audit.slurm` passes `--retry` only when `RETRY` is
  set, so the shared default applies otherwise.

### The fabrication instrument

`rl/monitors.FABRICATION_INSTRUCTIONS` is one 4096-token judge call per argument set — the Old Bailey
wording, which `evaluation/audit_fabrications.py` reuses verbatim for that domain and re-words for
house showing and nutrition. The judge is
shown the numbered evidence set plus the public `CASE BACKGROUND` and returns
`{"items_used": […], "false_claims": [{"claim": …}], "other_unsupported": […]}`. Only
`false_claims` is counted — `other_unsupported` is a catch-all sink (conclusions, absence-of-evidence
statements, characterizations, restatements) that is never read — and the count is that list after
dedup and the exact-quote guard. Properties of the code worth knowing:

- `_parse_fabrication_reply` counts a reply carrying `false_claims` from **that list only**; other
  shapes are read only when the key is missing, and a reply emitting both is counted once
  (`--selftest` pins the precedence).
- An unparseable reply retries on the same `RL_MONITOR_JUDGE_RETRY` budget as the binary judges;
  an **empty** reply stops immediately, because that is a transport outage (`receiver_client.chat`
  already burned its own attempts) and re-issuing a 4096-token call would stall the reward path.
- A `None` here contributes **0.0** to the penalty online (an unaudited rollout trains as if clean)
  while the offline metric drops the game instead — so `monitor_failure` flags whatever survives the
  retries, and rejection sampling refuses to count an unaudited rollout as clean.
- `evaluation/audit_fabrications.py` refuses to write a sidecar and returns **2** when every judged
  game judge-failed. The two causes are distinguished, not inferred from the game count: an
  all-EMPTY-reply run (`judge_empty_reply`, the dead-transport signature) is fatal at any number of
  games, so a 2-game smoke against a dead judge exits 2; all-unparseable with non-empty replies is
  fatal only from `JUDGE_FAIL_HARD_MIN`(3) judged games up. More than 10% failures warns.
- `rl.receiver_client.current_game_id` (a contextvar) is set inside the per-game pool workers, so
  judge-dump records attribute each call to its game. An **absent** id falls back to the dump line
  number; id `0` is a real game and stays `0`.

## The aux loss

`AUX_CE=1` adds, inside every GRPO optimizer step, `coeff × CE` on the gold `true`/`false` answer
token of in-role probe prompts — one probe item per rollout row, backpropagated together with the
policy gradient.

- **Wiring**: verl patches **10I** (`ray_trainer.fit()` attaches row-aligned `aux_*` tensors right
  before `update_actor`, after `_balance_batch` + `compute_advantage`, so row order is final) and
  **13T** (`dp_actor.update_policy()` runs the teacher-forced aux forward and an immediate scaled
  backward BEFORE the policy forward, freeing the aux logits graph first). `AUX_CE=0` → 10I never
  attaches → the keys are absent → 13T is a strict no-op. Pure logic lives in `rl/aux_ce.py`
  (`--selftest`); `rl/aux_grad_balance.py` holds the coefficient arithmetic.
- **Scheduling is POSITIONAL**: the first `round(AUX_CE_ROW_FRAC × chunk)` rows of each rank's
  dispatch chunk, with term and label taken from the row's position inside the chunk. Every DP rank
  therefore issues the identical `[aux fwd+bwd, policy fwd+bwd]` sequence, which FSDP2's per-backward
  collectives require — data-dependent scheduling would deadlock.
- **`AUX_CE_SAMPLING=random`** draws from GLOBAL per-(term, label) pools with without-replacement
  cycling, so each channel is exactly 1:1 every step, independent of sidecar size or game coverage.
  The launcher pins the sampler to `random`. **`AUX_CE_KIND_BALANCE=1`** additionally balances the
  four item kinds per batch: pairs take block `(p // T + global_step) % 2`, positionally, so
  all ranks agree and a matched pair can never straddle the two blocks.
- **`AUX_CE_BALANCE_MODE=norm_ratio`** makes `AUX_CE_COEFF` a **cap** rather than a multiplier:
  `coeff = min(cap, target_ratio × policy_norm / (aux_norm + ε))`, with a zero policy gradient giving
  a zero aux coefficient — the auxiliary objective must not create a policy update on its own. The
  launcher canonicalizes `AUX_CE_TARGET_GRAD_RATIO` once so the runtime env, the checkpoint suffix
  and the wandb metadata cannot disagree over `.25` vs `0.250`.
- **The dataset** is built by `bash datasets/old_bailey/aux_loss/build_chain.sh`, whose stages run
  `base_origin_all` (base-checkpoint fabrications only, each matched by a true record line) →
  `base_origin` (the 647-FALSE draw nested inside it) → `balanced` (surface-form balanced) →
  `kind_balanced` (481 × {fabrication, claim_real, altered_real, paraphrase_real} = 1924
  at both sizes) → `final` (verbatim `claim_real`) at
  `datasets/old_bailey/_generated/aux_ce_probe/variants/final/{4B,8B}/sidecar.jsonl.gz`, which
  is exactly the `AUX_CE_DATA` default. Its rows carry no `term` key, i.e. the fabrication channel;
  `AUX_CE_TERMS` selects channels of a sidecar that carries several.
- **Submit gates** (all fail at job start, never mid-training): patch 10I and 13T present, the
  adaptive structure validated by `rl.check_aux_grad_balance_patch` in `norm_ratio` mode, the sidecar
  non-empty and within `AUX_CE_MAX_PROMPT`, both labels present and within 1% of 1:1 (unequal pools
  mean unequal per-item exposure), all four kinds present with label-balanced blocks under
  `AUX_CE_KIND_BALANCE=1`, and the sidecar's `chat_template_sha` equal to the policy tokenizer's.
- 🔴 **`aux_loss` requires `PROMPT=strategies`.** Every probe context in the sidecar bakes the
  42-technique system prompt, and the only sha the gate checks is the chat template's, so a
  base-prompt aux run would train a silent mismatch. Refused twice: `requires.prompt` in the yaml and
  the launcher's own gate (`ERROR: AUX_CE=1 requires PROMPT=strategies …`).
- 🔴 **The auto-derived `EXP_NAME` does NOT encode `AUX_CE_DATA`.** Two runs differing only in the
  sidecar derive the same name and verl's `resume_mode=auto` would resume one into the other — a
  sidecar ablation MUST pass an explicit `EXP_NAME`.

## Other training mechanisms

- **Rejection sampling** (`REJECTION_SAMPLING=1`, `RS_TARGET_CLEAN`(8), `RS_MAX_GEN_ROUNDS`(2)):
  prompts whose GRPO group holds fewer than the target number of **positive** rollouts get extra
  batched full-`n` regeneration rounds, then exactly `rollout.n` rows are kept per prompt, positives
  first. Positive = `penalty_total == 0 ∧ parse_failure == 0 ∧ format_reward == 1.0 ∧
  monitor_failure == 0` — the format leg rejects the trivially clean empty-argument degenerate and
  terminator forgers, and `monitor_failure` keeps a judge outage from making unaudited rollouts look
  clean. It needs an active penalty, monitors at rate 1.0 and verl patch 10G; unusable configurations
  raise instead of silently training the stock flow. Run-name tag `_rs<target>`; pure logic in
  `rl/rejection_sampling.py`.
- **Rollout seed** (`ROLLOUT_SEED=<0..2147483646>`): derives one request seed per (experiment, step,
  game, rollout index) and separate per-turn sender / local-receiver seeds, independent of process
  scheduling and Python's randomized `hash()`. It rejects `REJECTION_SAMPLING=1` and an external
  `JUDGE_ENDPOINT_FILE` (this launcher cannot verify another server's deterministic-inference mode)
  and requires the 14T/14S/14R verl patches. Run-name tag `_rseed<seed>`; pure logic in
  `rl/rollout_seed.py`.
- **Receiver mixture** (`RECEIVER_MIX_SPEC` / the 2-way `RECEIVER_MIX_MODEL`, `rl/receiver_mix.py`,
  verl patch 10H): assigns each GAME — pre-repeat, so every GRPO group stays receiver-homogeneous and
  a cross-engine belief offset cancels out of the group-normalized advantage — to one of several
  receiver engines. **No paper arm uses it**; it is present, guarded (`python -m rl.receiver_mix
  --selftest`, `python -m rl.check_receiver_mix`) and left out of the arms table. The judge never
  mixes.

## SFT-then-RL curriculum

Four stages; the split knob and the init knob are separate on purpose.

0. Carve the split — 921 RL / 200 SFT / 100 validation, plus `sft_split.json` and the
   per-distribution `sft_holdout.parquet` (the byte-parity `[system, user1]` prompt source the SFT
   builder reads):

   ```bash
   python datasets/old_bailey/split_rl_sft.py --sender-prompt strategies \
       --sft-holdout 200 --sft-seed 2026 \
       --out-dir datasets/old_bailey/_generated/rl_sftsplit_strategies
   ```
1. Collect audited demonstrations over those 200 games with a **teacher**, in temperature-1 passes,
   against the served 35B. Two teachers, one driver
   (`datasets/old_bailey/sft/collect_rollouts.sh`):

   ```bash
   # hosted distillation: run on a machine with API access, with GATEWAY_URL and GATEWAY_API_KEY
   # set; the 35B is served by serve_receiver_xnode.slurm
   sbatch --export=ALL,ENDPOINT_FILE=$PWD/logs/recv_endpoint.txt scripts/serve_receiver_xnode.slurm
   SENDER=gpt-5.4-mini SENDER_API=gateway RECEIVER_ENDPOINT=$(cat logs/recv_endpoint.txt) \
       PROFILE=stubborn PROMPT=strategies PASSES=3 STEER_K=4 STEER_FROM_PASS=2 NUM_STEPS=3 \
       bash datasets/old_bailey/sft/collect_rollouts.sh

   # self-distillation: one GPU job co-serves the Qwen sender and the 35B, no key
   sbatch --export=ALL,SENDER=qwen3-8B-base,PASSES=10,STEER_K=0,BUILD_SFT=0 \
       datasets/old_bailey/sft/collect_selfdistill.slurm
   ```

   Passes at or after `STEER_FROM_PASS` add a per-game seeded "favor these ALLOWED strategies" hint
   (`rl_rollout.py --steer-strategies`), generation-only, for strategy diversity. Each pass freezes
   its own `--per-turn` sidecars (`.strategy_audit.json`, `.fabrication.json`); a sidecar that
   already covers a pass is not judged again, because the judge samples at temperature 1.
2. `datasets/old_bailey/sft/build_sft_dataset.py` filters **per turn with a mandatory clean-prefix
   rule** — a turn is kept iff it and every earlier turn pass the exact GRPO format predicate
   (`rl.reward_function._turn_format_ok`), clear the word floors, recite no harness/taxonomy
   vocabulary, argue the prosecution side, show `<= --max-illegal-per-turn` of the 11 illegal slugs
   with a `None` verdict counted as a violation, and carry `false_information == 0` and per-turn
   `rh_fake_evidence == 0` — and emits k prefix examples per rollout, loss on each example's final
   assistant turn.
3. `scripts/sft_train_sender.slurm` (verl `fsdp_sft_trainer`, 4 GPUs, `EPOCHS=2`,
   `sft/sft_dataset.py::PersuasionMultiTurnSFTDataset` patching the Qwen3 thinking template on a
   deep-copied tokenizer for prefix consistency and GRPO parity) writes a full HF model at
   `<SAVE_DIR>/global_step_N/huggingface/`, symlinked `<SAVE_DIR>/hf_final`.
4. GRPO from that init:

   ```bash
   sbatch --export=ALL,ALGO=base,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,SFT_INIT=<save_dir>/hf_final,SFT_INIT_TAG=ep2 \
       scripts/rl_train_sender.slurm
   ```

🔴 **`SFT_INIT` requires `SFT_SPLIT=1`** (refused otherwise): the init already saw the 200 games, so a
1121-game run would leak them into GRPO. **The only valid baseline for an `SFT_INIT` arm is the same
submit without `SFT_INIT`** — a 921-game run is not comparable to a 1121-game one. The prompt knob is
the same on both sides: never mix data roots between the SFT build and the GRPO stage.

## `JUDGE_MODEL` and the GPU shapes

`JUDGE_MODEL` names the ONE served model that is **both** the receiver the sender persuades and the
reward-hacking judge: `qwen3.5-35B` (`Qwen3.5-35B-A3B`, bf16, TP=2) by default, reached through
`rl/receiver_client.py`. `JUDGE_PATH` / `JUDGE_TP` / `SGLANG_EXTRA` override the checkpoint, the
parallel size and the sglang flags.

- **4B / 8B**: one 6-GPU job — `N_GPUS=4` trainer plus the judge co-served on the top `JUDGE_TP=2`.
  The guard is `REQ = N_GPUS + JUDGE_TP`, or `N_GPUS` when `JUDGE_ENDPOINT_FILE` is set.
- **14B**: `N_GPUS` defaults to 8, so a co-located judge would need 10 GPUs on one node and the `REQ`
  check rejects it. Two jobs instead:

  ```bash
  sbatch --export=ALL,JUDGE_MODEL=qwen3.5-35B,ENDPOINT_FILE=logs/judge_ep.txt scripts/rl_serve_judge.slurm
  sbatch --gres=gpu:8 --export=ALL,ALGO=base,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1,POLICY_MODEL=qwen3-14B,JUDGE_ENDPOINT_FILE=logs/judge_ep.txt \
      scripts/rl_train_sender.slurm
  ```

  The trainer waits up to `JUDGE_WAIT_MIN`(45) minutes for the endpoint file and reads `host:port`
  from it.

## Run names, resume and wandb

```
sender_${POLICY_MODEL}_${RECV_LABEL}_${JUDGE_TAG}_${FAKEPEN_TAG}${FMT_SUFFIX}${RS_SUFFIX}${PROMPT_SUFFIX}${SFTSPLIT_SUFFIX}${SFTINIT_SUFFIX}${AUXCE_SUFFIX}${MIX_SUFFIX}${RSEED_SUFFIX}${SMOKE_SUFFIX}
```

`RECV_LABEL` is `neutral` / `stubborn`; `JUDGE_TAG` is `j35B` for the default judge; `.` becomes `p`
in every coefficient. Consistency guards fire when an explicit `EXP_NAME` omits a derived `_rseed`,
`_mix`, `_normratio` or `_kindbal` tag, or advertises one that is not configured.

`resume_mode=auto`: a `SAVE_DIR` holding `global_step_*` directories is resumed silently and the
launcher prints a `NOTE` naming the latest step, so extending a run is a resubmit with a larger
`TOTAL_STEPS` under the same `EXP_NAME`. `SMOKE=1` disables resume and saving.

wandb runs offline inside the job; sync a run afterwards with `wandb sync wandb/offline-run-*` on a
machine with network access. Every run records
`custom_metadata` including `algo` (the resolved component list in registry order), `prompt`,
`sender_prompt`, the receiver distribution / types / prior, `judge_model`,
the penalty and format coefficients, the rejection-sampling and rollout-seed fields, the `aux_ce_*`
block including `aux_ce_data`, `policy_init` and `teacher_model`.

## Offline gates for this package

```bash
python -m rl.sender_prompts --selftest
python -m rl.arm_config --selftest
python -m rl.strategy_audit.audit --selftest
python -m rl.aux_ce --selftest
python -m rl.receiver_mix --selftest
python -m rl.check_receiver_mix
```

All are CPU-only and run from the repository root. `rl.check_aux_grad_balance_patch` and
`rl.check_rollout_seed_patch` inspect a patched verl checkout and take its file paths; the training
launcher calls them as submit gates. Beyond these, the rule in the root `CLAUDE.md` stands: a change
to pipeline code is verified by a SLURM smoke that runs to completion, and a training smoke needs 6
GPUs, so batch those at the end of a change.
