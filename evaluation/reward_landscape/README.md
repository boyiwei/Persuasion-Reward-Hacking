# The strategy-level reward landscape

Which persuasion techniques move a juror's belief? This package plays the Old Bailey game with a stock
sender under 44 different prompts, audits every argument for all 42 techniques of the taxonomy, and
estimates, per technique, the adjusted correlation between "the sender used this technique in this
game" and the juror's final P(guilty). It produces the paper's `fig_strategy_belief_correlation`.

## Protocol

The two stock senders, Qwen3-4B-Instruct-2507 and Qwen3-8B, play the 100 held-out Old Bailey games
(the paper's test set, 3 rounds each) against the stubborn juror profile (prior 0.1) of the locally
served Qwen3.5-35B-A3B, once per prompt arm. The arm is the `--sender-prompt` spec of
`evaluation/rl_rollout.py`:

| arm | `--sender-prompt` spec | prompt |
|---|---|---|
| `base` | `base` | the plain prosecutor role prompt |
| `strat` | `strategies` | the role prompt plus the 42-technique `ALLOWED` / `FORBIDDEN` guide |
| `only_<slug>` | `single_strategy:<slug>` | the role prompt plus one line allowing exactly that technique (42 arms) |

2 senders x 44 arms x 100 games = 88 cells, 8,800 games; only the prompt differs between cells. The
analysis keeps games with an audited argument and a valid final belief, and prints that count. The
juror is also the audit judge, so `rollout_sweep.slurm` rejects any `RECEIVER` other than
`qwen3.5-35B`, and one judge instance audits every cell (fabrication audit, then the 42-way technique
audit).

Per game, `x_k` = technique *k* present in the sender's arguments and `y` = the juror's final
P(guilty). The reported `r_adj` is their Pearson correlation after cell fixed effects (arm x sender)
and case fixed effects (the Old Bailey case) are partialled out of `y` and of each `x_k` by
alternating demeaning. This is the paper's adjusted point-biserial correlation with receiver belief
shift: at a fixed prior of 0.1, final belief and belief shift differ by a constant. The 42-way judge
is sampled, so a rerun reproduces the ranking, not the same digits.

## Running it

Everything reads and writes under `RESULTS_ROOT` (default `<repo>/experiments/results`;
`--results-root` for the Python entry points). Submit from the repo root.

```bash
export RESULTS_ROOT=/path/to/results          # optional; this is the default

# 1. rollouts: one job per sender, 3 GPUs (sender TP=1 + juror TP=2), ~a day for 44 arms
ONLY=$(.venv/bin/python -c "from rl.strategy_audit.taxonomy import slugs; print(' '.join('single_strategy:'+s for s in slugs()))")
for S in qwen3-4B-instruct-base qwen3-8B-base; do
  sbatch --export=ALL,SENDER=$S,PROMPTS="base strategies $ONLY" \
      evaluation/reward_landscape/rollout_sweep.slurm
done

# 2. manifest: the 88 cells, constructed forward and asserted present
.venv/bin/python evaluation/reward_landscape/build_manifest.py

# 3. the audit: one judge instance over the whole manifest, 4 GPUs (35B tp=2 x dp=2)
sbatch evaluation/reward_landscape/reaudit_all.slurm

# 4. gate + estimate, then draw
bash evaluation/reward_landscape/run.sh
.venv/bin/python evaluation/reward_landscape/plot_belief_correlation.py \
    --stats "$RESULTS_ROOT/reward_landscape/belief_correlation_stats.json"
```

Step 1 needs 80 GB cards and plays the games named by
`datasets/old_bailey/_generated/rl/bayesian/rl_validation.parquet` (written by
`python datasets/old_bailey/split_rl_sft.py`; `VAL_PARQUET` binds another split). Steps 3 and 4 read
the cell list from `MANIFEST` (default: step 2's manifest). Step 3 takes its data-parallel width from
`--gres`, so `sbatch --gres=gpu:2 --mem=96G --cpus-per-task=8 evaluation/reward_landscape/reaudit_all.slurm`
runs it on 2 GPUs. Steps 2 and 4 are CPU-only. Interpreters: `rollout_sweep.slurm` reads `SGLANG_VENV`
/ `SGLANG_PY` / `INFER_PYTHON`, `reaudit_all.slurm` reads `RL_VENV`, `run.sh` reads `PYTHON` (all
default to `$REPO/.venv`), and each ignores the others' knobs.

`--senders` / `--jurors` (`build_manifest.py`, `analyze_belief_correlation.py`) and `PROMPTS` /
`RECEIVERS` (the rollout job; juror profiles, not model ids) widen the grid to other sender sizes and
the neutral juror. The figure's default panel, `qwen4b8b_stub`, is estimated within its own rows
whatever else is in the grid.

## Outputs

| path under `$RESULTS_ROOT` | written by |
|---|---|
| `old-bailey/<sender>/<profile>_oldbailey_rlrollout_val_sender-<stem>__recv_qwen3.5-35B.json` | step 1 (`+ .json.metrics.json`) |
| `reward_landscape/manifest.tsv` | step 2 |
| `old-bailey/_reward_landscape_audit/j<jobid>/by_result/<key>.{fabrication,strategy_audit}.json` | step 3, published as `.../final` |
| `reward_landscape/belief_correlation_stats.json` | step 4 |
| `reward_landscape/fig_strategy_belief_correlation.{png,pdf,json}` | step 4 |

A cell is keyed `<sender dir>__<profile>__<arm>` with `arm` in `{base, strat, only_<slug>}`.
`run.sh` runs `check_inputs.py` (the provenance gate over the audit directory) and then
`analyze_belief_correlation.py`; the audit job gates each cell with `gate_audit.py`, and
`audit_source.py` says where a cell's files live.

## Smoke and checks

Two games, one round, two cells. No technique clears the `MIN_POS` guard with 2 games, so the plot
exits with "no estimable technique"; that is the expected result.

```bash
SMOKE_MANIFEST=$RESULTS_ROOT/reward_landscape/manifest_smoke.tsv
SMOKE_AUDIT=$RESULTS_ROOT/old-bailey/_reward_landscape_audit_smoke

sbatch --export=ALL,SENDER=qwen3-4B-instruct-base,SMOKE=1 evaluation/reward_landscape/rollout_sweep.slurm
.venv/bin/python evaluation/reward_landscape/build_manifest.py \
    --senders qwen3-4B-instruct-base --prompts strategies single_strategy:anchoring \
    --out "$SMOKE_MANIFEST"
sbatch --export=ALL,MANIFEST=$SMOKE_MANIFEST,OUTROOT=$SMOKE_AUDIT \
    evaluation/reward_landscape/reaudit_all.slurm
MANIFEST=$SMOKE_MANIFEST AUDIT_DIR=$SMOKE_AUDIT/final/by_result \
    bash evaluation/reward_landscape/run.sh
```

The audit job also takes `DRY_RUN=1` (print both passes' argv and run nothing; on a machine without GPUs,
pass `NGPU=4`), `ONLY=` (keys by substring) and `SMOKE=1` (one arm across the manifest). A subset run never
publishes `final`. The offline checks need no GPU or network:

```bash
.venv/bin/python evaluation/reward_landscape/build_manifest.py --selftest   # 44 arms, 88 keys, key round-trip
.venv/bin/python -m rl.sender_prompts --selftest                            # the prompt specs and their names
.venv/bin/python -m rl.strategy_audit.audit --selftest                      # the 42-way judge, online == offline
.venv/bin/python evaluation/audit_fabrications.py --selftest                # the fabrication judge's wordings
```
