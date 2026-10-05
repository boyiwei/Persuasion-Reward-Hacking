# Old Bailey

The primary environment: annotated Old Bailey trials played as a three-round persuasion game, with the
sender as prosecutor and the receiver as juror. [`sft/`](sft/README.md) builds the SFT demonstration
corpora and [`aux_loss/`](aux_loss/README.md) the auxiliary-loss sidecar. Run scripts under
`datasets/` by path from the repository root, so this directory does not shadow the HuggingFace
`datasets` library.

## The template

`processed/full/old_bailey_revised_independent_full.json` is a JSON list of 1,225 games, each with an
integer `id`, a `params` block, and empty `rounds` / `responses` / `complete` fields that a rollout
fills. `params.public` is the prose both players read (`game_background`, `state_space`,
`prior_belief`, `action_space`, `sender_utility`, `receiver_utility`). `params.private.information` is
the evidence, one string per item:

```
[EVIDENCE ev1]
Description: Ann Griffiths testified that George Gowens approached her, accused her of ...
Prosecution Strength: 0.8
Prosecution Reasoning: This is the core testimony from the victim establishing the assault ...
Defense Strength: 0.2
Defense Reasoning: The testimony mentions a preceding event, which hints at a personal dispute ...
```

Games hold 1 to 13 items (mean 5.1); four hold none. `rl.evidence.clean_evidence` keeps only the
handle and the `Description:` line before evidence reaches a sender prompt. The strengths and
reasonings are ground truth for the reward function, the monitors and the fabrication audit.

## The split

`split_rl_sft.py` drops the four empty games (1,221 playable), shuffles them with
`random.Random(--seed)` (42), and takes the first `--val-size` (100) as the held-out validation set,
identical for every sender prompt and SFT seed. `--sft-holdout 200 --sft-seed 2026` samples 200 SFT
games from the remaining 1,121 with an independent RNG, leaving 921 RL / 200 SFT / 100 validation.
Rows are written once per receiver distribution: `bayesian` (neutral juror, prior P(guilty)=0.5) and
`stubborn` (prior 0.1, with presumption-of-innocence `prior_belief` prose). The sender prompt is baked
into every row, so each `--sender-prompt` spec needs its own `--out-dir`.

```bash
python datasets/old_bailey/split_rl_sft.py                        # base prompt, 1121 / 100
python datasets/old_bailey/split_rl_sft.py --sft-holdout 200 --sft-seed 2026 \
    --out-dir datasets/old_bailey/_generated/rl_sftsplit            # base prompt, 921 / 200 / 100
python datasets/old_bailey/split_rl_sft.py --sender-prompt strategies \
    --sft-holdout 200 --sft-seed 2026 \
    --out-dir datasets/old_bailey/_generated/rl_sftsplit_strategies # guide prompt, 921 / 200 / 100
```

Other flags: `--input` (the template), `--end-index` (use only `games[:n]`), `--val-size`, `--seed`.
A root holds `<dist>/rl_train.parquet` and `<dist>/rl_validation.parquet` for `dist` in
`{bayesian, stubborn}`, plus `<dist>/sft_holdout.parquet` (the 200 SFT games, `split="sft"`) and
`sft_split.json` (`rl_train_ids` / `sft_train_ids`, seeds, counts) when a holdout is carved. Each row
(built by `rl/game_rows.py`) carries the sender's initial chat over the cleaned evidence, `extra_info`
with the game id, split, `n_pros_favoring` and the slim receiver payload, and the annotated `payload`
the reward function and monitors read.

`scripts/rl_train_sender.slurm` picks the root from its `PROMPT` and `SFT_SPLIT` knobs and runs
`split_rl_sft.py` itself when the parquets are missing; `DATA_DIR` overrides it.

| root under `datasets/old_bailey/_generated/` | launcher knobs |
|---|---|
| `rl/` | `PROMPT=base` (`SFT_SPLIT=0`) |
| `rl_sftsplit/` | `PROMPT=base SFT_SPLIT=1` |
| `rl_sftsplit_strategies/` | `PROMPT=strategies SFT_SPLIT=1` |
| `rl_only_<slug>/`, `rl_sftsplit_only_<slug>/` | `PROMPT=single_strategy:<slug>` |
| `rl*_smoke/` | any of the above with `SMOKE=1` |
