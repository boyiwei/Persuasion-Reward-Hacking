#!/usr/bin/env python3
"""Split the Old Bailey game templates into the RL, SFT and validation sets, and write the parquets.

Split:
  1. Read processed/full/old_bailey_revised_independent_full.json (1225 cases) and drop games with
     no evidence, leaving 1221.
  2. Shuffle with random.Random(--seed=42) and take the first --val-size (100) as validation
     (rl/game_rows._seeded_val_idx), so validation is identical for every sender prompt and SFT seed.
  3. With --sft-holdout 200, an independent random.Random(--sft-seed=2026) samples 200 of the sorted
     train ids as SFT-train: 921 RL / 200 SFT / 100 val. Writes <out-dir>/sft_split.json and
     <out-dir>/<dist>/sft_holdout.parquet, the [system, user1] prompt source for
     datasets/old_bailey/sft/build_sft_dataset.py.

Rows come from rl/game_rows.py for each receiver distribution (bayesian, stubborn). The sender
prompt is baked into the parquet, so each --sender-prompt spec needs its own --out-dir.

Usage (repo root, repo .venv):
    python datasets/old_bailey/split_rl_sft.py                  # _generated/rl, 1121 / 100
    python datasets/old_bailey/split_rl_sft.py --sender-prompt strategies --sft-holdout 200 \
        --sft-seed 2026 --out-dir datasets/old_bailey/_generated/rl_sftsplit_strategies

scripts/rl_train_sender.slurm builds the root its knobs select by calling this script.
"""
import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from agents.agent_quality import AgentQuality  # noqa: E402
from agents.model.model import ModelAPI  # noqa: E402
from rl.game_rows import (  # noqa: E402
    _DEFAULT_PRIOR,
    _PRIOR_BELIEF_BY_DIST,
    _PRIOR_BY_DIST,
    _n_pros_favoring,
    _row,
    _seeded_val_idx,
    _spec_for,
)
from rl.sender_prompts import (  # noqa: E402
    PromptSpecError,
    parse_spec,
    resolve_sender_prompt,
)

_DEFAULT_INPUT = _REPO / "datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json"
_DEFAULT_OUT = _REPO / "datasets/old_bailey/_generated/rl"


def build(distribution: str, games: list, agent: AgentQuality, val_size: int,
          seed: int, sft_exclude_ids: frozenset = frozenset()) -> tuple:
    """Return (train_rows, val_rows, sft_rows) for distribution in {'bayesian','stubborn'}.

    Train rows keep template order (the trainer shuffles). `sft_exclude_ids` are train-side games
    moved to `sft_rows` (split="sft")."""
    val_idx = _seeded_val_idx(len(games), val_size, seed)
    prior = _PRIOR_BY_DIST.get(distribution, _DEFAULT_PRIOR)
    prior_belief = _PRIOR_BELIEF_BY_DIST.get(distribution)

    train_games = [g for i, g in enumerate(games) if i not in val_idx]
    val_games = [g for i, g in enumerate(games) if i in val_idx]
    sft_games = [g for g in train_games if g["id"] in sft_exclude_ids]
    if sft_exclude_ids:
        train_games = [g for g in train_games if g["id"] not in sft_exclude_ids]

    train_rows = [_row(g, _spec_for(distribution), agent, "train", prior, prior_belief)
                  for g in train_games]
    val_rows = [_row(g, _spec_for(distribution), agent, "validation", prior, prior_belief)
                for g in val_games]
    sft_rows = [_row(g, _spec_for(distribution), agent, "sft", prior, prior_belief)
                for g in sft_games]
    return train_rows, val_rows, sft_rows


def main():
    ap = argparse.ArgumentParser(description="Build Old Bailey Sender-RL parquets")
    ap.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--end-index", type=int, default=None, help="use only games[:end_index]")
    ap.add_argument("--val-size", type=int, default=100, help="held-out validation games")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sender-prompt", default="base", metavar="SPEC",
                    help="which sender prompt to bake into every row: base (default) | strategies "
                         "(the 42-technique ALLOWED/FORBIDDEN guide) | single_strategy:<slug> "
                         "(one allowed technique). rl/sender_prompts.py owns the spec, and each "
                         "spec needs its OWN --out-dir: the prompt is baked into the parquet.")
    ap.add_argument("--sft-holdout", type=int, default=0,
                    help="SFT-then-RL curriculum: carve N games out of the TRAIN side (never the "
                         "validation set) as the SFT-train set. Train parquets shrink by N; the "
                         "held-out games are written per dist as sft_holdout.parquet (split='sft') "
                         "and the id split as <out-dir>/sft_split.json.")
    ap.add_argument("--sft-seed", type=int, default=2026,
                    help="seed for the SFT holdout sample (independent of --seed, which fixes the "
                         "train/val shuffle -- validation stays byte-identical for any sft-seed)")
    args = ap.parse_args()

    # Parse first so a spec typo aborts before any split artifact is written.
    try:
        spec = parse_spec(args.sender_prompt)
    except PromptSpecError as e:
        ap.error(str(e))

    with open(args.input) as f:
        raw = json.load(f)
    # Drop games with no evidence (None or empty information list).
    games = [g for g in raw if (g.get("params", {}).get("private") or {}).get("information")]
    dropped = len(raw) - len(games)
    if args.end_index is not None:
        games = games[:args.end_index]
    val_size = min(args.val_size, max(0, len(games) // 5))
    print(f"[prepare] {args.input.name}: {len(raw)} cases, {dropped} dropped (zero evidence), "
          f"{len(games)} kept (val_size={val_size})")

    # SFT holdout: train side only, own rng, ids sorted so template order does not matter.
    sft_exclude = frozenset()
    if args.sft_holdout > 0:
        val_idx = _seeded_val_idx(len(games), val_size, args.seed)
        train_games = [g for i, g in enumerate(games) if i not in val_idx]
        val_ids = {games[i]["id"] for i in val_idx}
        k = min(args.sft_holdout, max(1, len(train_games) // 2))  # clamp for tiny smokes
        sft_rng = random.Random(args.sft_seed)
        sft_ids = sft_rng.sample(sorted(g["id"] for g in train_games), k)
        sft_exclude = frozenset(sft_ids)
        assert not (sft_exclude & val_ids), "SFT holdout leaked into the validation set"
        from collections import Counter
        by_id = {g["id"]: g for g in games}
        hist = Counter(_n_pros_favoring(by_id[i]) for i in sft_exclude)
        print(f"[prepare] sft holdout: {k} of {len(train_games)} train games (sft-seed "
              f"{args.sft_seed}); n_pros histogram: "
              + ", ".join(f"{n}:{hist[n]}" for n in sorted(hist, reverse=True)))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        split_art = {
            "input": str(args.input),
            "seed": args.seed,
            "val_size": val_size,
            "sft_seed": args.sft_seed,
            "sft_holdout": k,
            "n_games_total": len(games),
            "sft_train_ids": sorted(sft_exclude),
            "rl_train_ids": sorted(g["id"] for g in train_games if g["id"] not in sft_exclude),
        }
        with open(args.out_dir / "sft_split.json", "w") as f:
            json.dump(split_art, f, indent=2)
        print(f"[prepare] sft split -> {args.out_dir / 'sft_split.json'}")

    sender_cfg = resolve_sender_prompt("old-bailey", spec)
    print(f"[prepare] sender prompt: {spec} "
          f"({len(sender_cfg.prompts.messages[0].content)} system chars)")
    agent = AgentQuality(sender_cfg, ModelAPI())
    for dist in ("bayesian", "stubborn"):
        train_rows, val_rows, sft_rows = build(dist, games, agent, val_size, args.seed,
                                               sft_exclude_ids=sft_exclude)
        assert len(train_rows) == len(games) - val_size - len(sft_exclude), \
            f"{dist}: train rows {len(train_rows)} != {len(games)} - {val_size} - {len(sft_exclude)}"
        out = args.out_dir / dist
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_rows).to_parquet(out / "rl_train.parquet")
        pd.DataFrame(val_rows).to_parquet(out / "rl_validation.parquet")
        sft_note = ""
        if sft_rows:
            pd.DataFrame(sft_rows).to_parquet(out / "sft_holdout.parquet")
            sft_note = f" / {len(sft_rows)} sft"
        print(f"[prepare] {dist}: {len(train_rows)} train / {len(val_rows)} val{sft_note} -> {out}")


if __name__ == "__main__":
    main()
