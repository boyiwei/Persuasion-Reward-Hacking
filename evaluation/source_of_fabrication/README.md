# Source of fabrication: the endorsement heatmap

Does a sender that fabricates evidence know the claim is false? This package separates the generation
step (which checkpoint produced the fabricated claim) from the probe step (which checkpoint is asked
whether that claim is true evidence) and measures every pair. It produces the paper's
`fig_endorsement_grid_bal100`.

## Protocol

Per policy size (4B, 8B) the design is a 3 x 3 grid: three generation checkpoints (step 0, the stock
model, and steps 50 and 100 of the base-prompt, no-penalty, stubborn-juror GRPO run) crossed with the
same three checkpoints as probers. Each generation checkpoint plays the 100 held-out Old Bailey games,
the served Qwen3.5-35B extracts its fabricated claims, and 100 fabrications per (size, generation
step) are drawn at random. Every prober answers every item under two probe templates: Direct Ask
(`out`, a fresh conversation with a neutral-analyst framing) and In-Role (`in_pre_role`, the judge
interrupts mid-game and demands an answer for the court record).

The cell value is the share of fabricated claims the prober accepts as true evidence. A column holds
the prober fixed and varies the evidence; a row holds the evidence fixed and varies the policy; the gap
between templates measures how context-dependent the awareness is. Answers are sampled at the
RL-rollout setting (temp 1 / top_p 1 / top_k -1) with draw #1 as the readout. Rollouts and judge are
sampled too, so a fresh run reproduces the design but not the exact cell values. The 35B is both the
stage-A juror and the fabrication judge, so `generate_items.slurm` rejects any `RECEIVER_NAME` other
than `qwen3.5-35B`.

## Prerequisites

Six checkpoints: the stock `Qwen3-4B-Instruct-2507` and `Qwen3-8B`, and the step-50 and step-100 merges
of the base-prompt stubborn run as `scripts/merge_verl_ckpt.slurm` names them
(`sender_qwen3-{4B,8B}_stubborn_j35B_fakepen0_fmt0p1_{gs50,gs100}`). `ckpt_paths.sh` resolves them
under `TC` (the stock models, default `$MODELS_DIR`, itself defaulting to `<repo>/models`) and `MERGED`
(the merged root, default `<repo>/experiments/results/rl/_em_merged`); override those or `PROFILE` if
yours live elsewhere. The held-out games come from
`datasets/old_bailey/_generated/rl/stubborn/rl_validation.parquet`, written by
`python datasets/old_bailey/split_rl_sft.py` (`VAL_PARQUET` binds another split).

## Stages

Everything is written under `SOF_ROOT` (default `<repo>/experiments/results/source_of_fabrication`),
which must sit outside this source directory. Submit the SLURM stages from the repo root; they serve
their own models and read `SGLANG_VENV` / `SGLANG_PY` / `INFER_PYTHON` (default `$REPO/.venv`).

### A + B: play, audit, build items (one job per generation cell, 3 GPUs, 80 GB cards)

```bash
for SIZE in 4B 8B; do for CKPT in base gs50 gs100; do
  sbatch --export=ALL,SIZE=$SIZE,CKPT=$CKPT evaluation/source_of_fabrication/generate_items.slurm
done; done
sbatch --export=ALL,SIZE=8B,CKPT=base,DRAW=d2 evaluation/source_of_fabrication/generate_items.slurm   # extra draw
```

Each job plays the held-out games with the base sender prompt against the stubborn juror, extracts
the fabricated claims (`audit_rollouts.py`, `AUDIT_PASSES` passes so a judge failure is retried), and
rebuilds the conversation each claim was made in (`build_probe_items.py`, `PER_GAME_CAP=3` claims per
game). The six primary cells land in `$SOF_ROOT/probe_items/primary/`. A cell whose pool falls short
of the target needs a second draw: `DRAW=d2` writes to `$SOF_ROOT/probe_items/d2/` with
`--emit-candidates` for B2 to pool. A cross-node `mkdir` lock lets a duplicate job for a cell exit 0.

### B2: draw the item set (CPU only)

```bash
.venv/bin/python evaluation/source_of_fabrication/build_item_set.py \
    --src-items $SOF_ROOT/probe_items/primary --raw-dir $SOF_ROOT/probe_items/d2 \
    --rollouts-dir $SOF_ROOT/rollouts --sizes 4B 8B --n-per-cell 100 --seed 2026 \
    --selftest --out-dir $SOF_ROOT/probe_items_union
```

Per (size, generation step) the pooled candidates are sorted by `item_id`, deduplicated on (game,
normalised statement), and N drawn uniformly with `random.Random(f"{seed}|{size}|{ckpt}")`, without
consulting any probe answer. The `control_real` / `control_distractor` items of each selected
context ride along. A cell with fewer than N candidates is an error. `--selftest` byte-compares the
composed messages against the source tables during a real draw; `--raw-dir` is optional.

### C: probe (one job per probe cell, 1 GPU)

```bash
for SIZE in 4B 8B; do for PCKPT in base gs50 gs100; do
  sbatch --export=ALL,SIZE=$SIZE,PCKPT=$PCKPT evaluation/source_of_fabrication/probe_cell.slurm
done; done
.venv/bin/python evaluation/source_of_fabrication/check_templates.py \
    --verify-dir $SOF_ROOT/answers_union --sizes 4B 8B      # after all six jobs
```

Each job serves one policy and answers every item under `PROBES` (default `out in_pre_role`; add
`in_pre_honest` for the honest-pause readout). Before serving it checks the probe prompt against its
pins and the prober's checkpoint fingerprint against the one stage A recorded. `check_templates.py`
is the cross-prober template gate.

### D: aggregate and plot (CPU only)

```bash
.venv/bin/python evaluation/source_of_fabrication/analyze.py \
    --ans-dir $SOF_ROOT/answers_union --items-dir $SOF_ROOT/probe_items_union \
    --src-items $SOF_ROOT/probe_items/primary $SOF_ROOT/probe_items/d2 \
    --rollouts-dir $SOF_ROOT/rollouts --out $SOF_ROOT/stats.json

.venv/bin/python evaluation/source_of_fabrication/plot_heatmap.py --stats $SOF_ROOT/stats.json \
    --outdir $SOF_ROOT/figures --name fig_endorsement_grid_bal100
```

`analyze.py` counts `<answer>true</answer>` as endorsement and `<answer>false</answer>` as detection,
and refuses to write stats unless every (prober x probe arm x origin) cell has its expected count
(`--allow-incomplete` overrides). `plot_heatmap.py` plots each cell's `hallucination_rate` to
`<name>.{png,pdf,json}`; the command above is the paper's figure, and `--show-n` adds per-cell n.

## Outputs

```
rollouts/        rollout__{size}__{ckpt}[__{draw}].json   + _ckpt__{size}__{ckpt}.json fingerprints
audits/          audit__{size}__{ckpt}[__{draw}].jsonl    one line per game, the false claims
probe_items/     primary/ and one dir per extra draw: incontext__*.jsonl, contexts__*.jsonl.gz,
                 _build_summary__*.json (+ candidates__* for an extra draw)
probe_items_union/  union__{size}.jsonl, contexts_union__{size}.jsonl.gz,
                 union_smoke__{size}.jsonl, _union_summary.json
answers_union/   {size}__P{prober}__{probe}.jsonl, _build_id__{size}, _ckpt__*.json,
                 template_check_{size}_{ckpt}.json
stats.json       the aggregate
figures/         the heatmap
logs/            server and fingerprint logs
```

## Smoke and checks

`SMOKE=1` runs 3 games per cell, one audit pass and 2 draws per item, writing `_smoke` copies of the
per-stage directories (`logs/` and `figures/` are shared). A 3-game cell rarely yields five
fabrications, so draw two items per cell, and run all three generation cells of the size:

```bash
for CKPT in base gs50 gs100; do
  sbatch --export=ALL,SIZE=8B,CKPT=$CKPT,SMOKE=1 evaluation/source_of_fabrication/generate_items.slurm
done
.venv/bin/python evaluation/source_of_fabrication/build_item_set.py \
    --src-items $SOF_ROOT/probe_items_smoke/primary --sizes 8B --n-per-cell 2 \
    --out-dir $SOF_ROOT/probe_items_union_smoke
for PCKPT in base gs50 gs100; do
  sbatch --export=ALL,SIZE=8B,PCKPT=$PCKPT,SMOKE=1 evaluation/source_of_fabrication/probe_cell.slurm
done
.venv/bin/python evaluation/source_of_fabrication/analyze.py \
    --ans-dir $SOF_ROOT/answers_union_smoke --items-dir $SOF_ROOT/probe_items_union_smoke \
    --src-items $SOF_ROOT/probe_items_smoke/primary --sizes 8B \
    --out $SOF_ROOT/stats_smoke.json --allow-incomplete
```

The smoke stops at `analyze.py` (which writes `meta.complete=false`), because `plot_heatmap.py` needs
every cell of both sizes. The offline checks need no GPU or network:

```bash
.venv/bin/python evaluation/source_of_fabrication/probe_prompts.py --selftest      # the probe text, pinned
.venv/bin/python evaluation/source_of_fabrication/build_probe_items.py --selftest  # attribution + splice
```
