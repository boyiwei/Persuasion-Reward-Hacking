#!/usr/bin/env python3
"""Build label-balanced subsets of an aux-CE probe sidecar by filtering rows (no GPU, no judge).

Dataset-size ablation. The parent's FALSE items come from the base and gs50/gs100 checkpoints,
which presupposes the checkpoints being trained. Variants answer: can FALSE items come from base
alone (--false-origin base), and how few are enough (--n-false N)? Both only make sense under
AUX_CE_SAMPLING=random (exact 1:1 from global per-label pools); under "game" sampling the FALSE
share tracks game coverage, not item count (see rl/aux_ce.py).

TRUE items are re-matched, since e.g. 3676 TRUE / 877 FALSE (4B) under 1:1 sampling would show each
FALSE item 4.19x as often. Each surviving game gets as many TRUE items as FALSE ones (control_real
in the fabrication context -> round-1 anchor -> control_real_aug), with any deficit spread
round-robin over games with spares, so TRUE and FALSE come from the same games.

Rows are copied verbatim in source order into GzipFile(mtime=0), so rebuilds are byte-identical.
Sampling uses string-seeded random.Random (salt `<seed>:<phase>:<tag>`, as in rl/aux_ce.draw).

  # base-origin FALSE only, TRUE matched 1:1  (4B: 877/877, 8B: 647/647)
  .venv/bin/python datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py --tag base_origin_all --false-origin base
  # a nested, smaller draw over those games                    (4B: ~220/220, 8B: ~162/162)
  .venv/bin/python datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py --tag base_origin_220 --false-origin base \
      --n-false 220 --n-false-8B 162 --nest-in base_origin_all
  # re-validate through the real loader and the submit gate's assertions
  .venv/bin/python datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py --tag base_origin_all --selfcheck-only
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    DEFAULT_SRC,
    DEFAULT_VARIANTS as DEFAULT_OUT,
    GENERATED,
    SIZES,
    builder_basename,
)

DEFAULT_PARQUET = GENERATED / "rl_sftsplit_strategies/stubborn/rl_train.parquet"
# TRUE-item preference (lower first). claim_real ranks with the control_real it replaces, so a
# mixed-schema parent does not reorder on the .get default. Rebalanced parents are still refused by
# _reject_rebalanced().
KIND_RANK = {"control_real": 0, "claim_real": 0, "control_real_aug": 1, "paraphrase_real": 2}
REBALANCED_KINDS = {"claim_real", "paraphrase_real", "altered_real"}
# The aux knobs load_state reads; --selfcheck sets them and puts the caller's values back.
_AUX_ENV_KEYS = ("AUX_CE", "AUX_CE_DATA", "AUX_CE_MAX_PROMPT", "AUX_CE_SAMPLING",
                 "AUX_CE_KIND_BALANCE")


def _reject_rebalanced(meta=None, rows=()):
    """Refuse a rebalanced parent with a clear error.

    Its rows are length/style-matched pairs that row-wise filtering would split. Its _meta.json
    also lacks top-level per_game_cap, so the meta check catches it before a bare KeyError.
    """
    # match on the builder's file name, not its path
    by_meta = bool(meta) and (
        builder_basename((meta.get("variant") or {}).get("builder", ""))
        == "rebalance_aux_ce_sidecar.py"
        or "per_game_cap" not in meta)
    if by_meta or any(r.get("kind") in REBALANCED_KINDS or r.get("pair_id") for r in rows):
        raise SystemExit(
            "[filter] this parent was produced by rebalance_aux_ce_sidecar.py "
            "(paired rows / rebalanced kinds). Row-wise filtering would split pairs and "
            "re-introduce the length confound the variant removes, and its _meta.json uses a "
            "different schema. Re-run rebalance_aux_ce_sidecar.py against a smaller parent "
            "instead of filtering this one.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path):
    """-> [(raw_line_without_newline, parsed_dict), ...] in source order."""
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                out.append((line.rstrip("\n"), json.loads(line)))
    return out


def select_false(rows, origins, n_target, seed, tag, nest_games=None):
    """Pick the gold-FALSE items. Returns (kept_item_ids, per_game_counts).

    `nest_games` (from a larger variant) restricts the pool so dose levels are nested."""
    by_game = collections.defaultdict(list)
    for _, r in rows:
        if r["label"] != "false":
            continue
        if origins and r["origin_ckpt"] not in origins:
            continue
        by_game[int(r["game_id"])].append(r["item_id"])
    for gid in by_game:
        by_game[gid].sort()                      # never depend on file order for the RNG
    games = sorted(by_game)
    if nest_games is not None:
        games = [g for g in games if g in nest_games]
        assert games, f"nesting left no games (nest set has {len(nest_games)})"

    if n_target is None or n_target >= sum(len(by_game[g]) for g in games):
        keep_games = games
    else:
        # Draw whole games (fewer cases, not fewer claims per case) until the budget is reached.
        rng = random.Random(f"{seed}:gamesample:{tag}")
        shuffled = games[:]
        rng.shuffle(shuffled)
        keep_games, total = [], 0
        for g in shuffled:
            if total >= n_target:
                break
            keep_games.append(g)
            total += len(by_game[g])
        keep_games.sort()

    kept, per_game = set(), {}
    for g in keep_games:
        kept.update(by_game[g])
        per_game[g] = len(by_game[g])
    return kept, per_game


def select_true(rows, per_game_false, seed, tag):
    """Pick exactly sum(per_game_false) gold-TRUE items from the FALSE-carrying games.

    Up to the FALSE count per game, then any residual round-robin over games with spares."""
    need_total = sum(per_game_false.values())
    avail = collections.defaultdict(list)
    for _, r in rows:
        if r["label"] != "true":
            continue
        gid = int(r["game_id"])
        if gid in per_game_false:
            avail[gid].append((KIND_RANK.get(r["kind"], 9), r["item_id"]))
    for gid in avail:
        avail[gid].sort()                        # kind priority, then item_id for determinism

    kept = set()
    spare = {}
    for gid, want in sorted(per_game_false.items()):
        pool = [iid for _, iid in avail.get(gid, [])]
        take = min(want, len(pool))
        kept.update(pool[:take])
        if len(pool) > take:
            spare[gid] = pool[take:]

    deficit = need_total - len(kept)
    if deficit > 0:
        rng = random.Random(f"{seed}:truefill:{tag}")
        order = sorted(spare)
        rng.shuffle(order)
        depth = 0
        while deficit > 0:
            progressed = False
            for gid in order:
                if depth < len(spare[gid]):
                    kept.add(spare[gid][depth])
                    deficit -= 1
                    progressed = True
                    if deficit == 0:
                        break
            if not progressed:
                break
            depth += 1
    assert deficit == 0, (
        f"cannot match TRUE 1:1 — need {need_total}, only {len(kept)} available in the "
        f"{len(per_game_false)} FALSE-carrying games")
    return kept


def build_size(size, args, src_root: Path, nest_games):
    src = src_root / size / "sidecar.jsonl.gz"
    assert src.is_file(), f"missing parent sidecar: {src}"
    rows = read_rows(src)
    _reject_rebalanced(rows=[r for _, r in rows])
    n_target = getattr(args, f"n_false_{size}", None) or args.n_false
    origins = set(args.false_origin.split(",")) if args.false_origin else None

    keep_false, per_game = select_false(rows, origins, n_target, args.seed, args.tag, nest_games)
    keep_true = select_true(rows, per_game, args.seed, args.tag)
    keep = keep_false | keep_true
    assert len(keep_false) == len(keep_true), (len(keep_false), len(keep_true))

    lines = [raw for raw, r in rows if r["item_id"] in keep]
    kinds = collections.Counter(r["kind"] for _, r in rows if r["item_id"] in keep)
    origin_f = collections.Counter(r["origin_ckpt"] for _, r in rows
                                   if r["item_id"] in keep_false)
    stats = {
        "n_items": len(lines), "n_true": len(keep_true), "n_false": len(keep_false),
        "by_kind": dict(kinds), "false_by_origin": dict(origin_f),
        "n_games_with_false": len(per_game),
        "n_games_with_true": len({int(r["game_id"]) for _, r in rows
                                  if r["item_id"] in keep_true}),
        "false_per_game_hist": dict(sorted(collections.Counter(per_game.values()).items())),
        "games_with_false": sorted(per_game),
    }
    return lines, stats


def write_variant(args, src_root: Path, out_root: Path, parent_meta: dict):
    _reject_rebalanced(parent_meta)          # before any key of the parent schema is touched
    out_dir = out_root / args.tag
    nest_games = None
    if args.nest_in:
        nest_meta = json.loads((out_root / args.nest_in / "_meta.json").read_text())
        nest_games = {sz: set(nest_meta["sizes"][sz]["games_with_false"]) for sz in SIZES}

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO), text=True,
                                capture_output=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    meta = {
        # Copied from the parent. Recomputing chat_template_sha would let a drifted tokenizer
        # pass the parity gate in rl/aux_ce.load_state.
        "schema_version": parent_meta["schema_version"], "arm": parent_meta["arm"],
        "split_file": parent_meta["split_file"], "per_game_cap": parent_meta["per_game_cap"],
        "max_prompt_tokens_cap": parent_meta["max_prompt_tokens_cap"],
        "source_git_commit": parent_meta["git_commit"], "git_commit": commit,
        "variant": {
            "tag": args.tag,
            "builder": "datasets/old_bailey/aux_loss/filter_aux_ce_sidecar.py",
            "argv": sys.argv[1:], "seed": args.seed,
            "filters": {"false_origin": args.false_origin or None, "n_false": args.n_false,
                        "n_false_4B": args.n_false_4B, "n_false_8B": args.n_false_8B,
                        "nest_in": args.nest_in or None},
            "true_matching": "per-game 1:1 (control_real -> r1 anchor -> control_real_aug), "
                             "residual spread round-robin; all TRUE from FALSE-carrying games",
            "parent_root": str(src_root),
        },
        "sizes": {},
    }

    for size in args.sizes:
        pm = parent_meta["sizes"][size]
        lines, stats = build_size(size, args, src_root,
                                  nest_games[size] if nest_games else None)
        d = out_dir / size
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "sidecar.jsonl.gz"
        if args.dry_run:
            print(f"[dry] {out_path}: {stats['n_items']} items "
                  f"({stats['n_true']}T/{stats['n_false']}F) over "
                  f"{stats['n_games_with_false']} FALSE games; origins={stats['false_by_origin']}")
            meta["sizes"][size] = {**stats}
            continue
        with gzip.GzipFile(out_path, "wb", mtime=0) as gz:
            for ln in lines:
                gz.write((ln + "\n").encode())
        m = dict(stats)
        # parent values, as above
        for k in ("tokenizer_path", "chat_template_sha", "max_prompt_tokens", "sources_sha256",
                  "audit_stats", "n_train_ids"):
            if k in pm:
                m[k] = pm[k]
        m["parent_sidecar_sha256"] = pm["sidecar_sha256"]
        m["parent_n_items"] = pm["n_items"]
        m["sidecar_sha256"] = _sha256(out_path)
        meta["sizes"][size] = m
        print(f"[write] {out_path}: {m['n_items']} items ({m['n_true']}T/{m['n_false']}F) over "
              f"{m['n_games_with_false']} FALSE games; origins={m['false_by_origin']}")

    if args.dry_run:
        return meta
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[done] meta -> {out_dir / '_meta.json'}")
    return meta


def selfcheck(args, out_root: Path):
    """Load each variant through rl.aux_ce.load_state and re-run the submit gate's assertions."""
    import pandas as pd
    from transformers import AutoTokenizer

    import rl.aux_ce as aux

    out_dir = out_root / args.tag
    meta = json.loads((out_dir / "_meta.json").read_text())
    train_ids = {int(e["index"]) for e in pd.read_parquet(args.train_parquet)["extra_info"]}
    # load_state reads the aux knobs from the environment; restored afterwards.
    _saved_env = {k: os.environ.get(k) for k in _AUX_ENV_KEYS}
    ok = True
    for size in args.sizes:
        path = out_dir / size / "sidecar.jsonl.gz"
        m = meta["sizes"][size]
        assert _sha256(path) == m["sidecar_sha256"], f"{path}: sha256 drifted from _meta.json"
        # Parent kinds are not the four paired kinds AUX_CE_KIND_BALANCE (default on) needs.
        os.environ.update({"AUX_CE": "1", "AUX_CE_DATA": str(path),
                           "AUX_CE_MAX_PROMPT": "8192", "AUX_CE_SAMPLING": "random",
                           "AUX_CE_KIND_BALANCE": "0"})
        tok = AutoTokenizer.from_pretrained(m["tokenizer_path"])
        st = aux.load_state(tok)
        n_t, n_f = len(st.items_by_label["true"]), len(st.items_by_label["false"])
        assert (n_t, n_f) == (m["n_true"], m["n_false"]), ((n_t, n_f), (m["n_true"], m["n_false"]))
        assert n_t == n_f, f"{size}: NOT balanced ({n_t}T/{n_f}F) — random sampling needs 1:1"
        # the submit gate's own checks (scripts/rl_train_sender.slurm)
        assert abs(n_t / n_f - 1.0) <= 0.01
        assert m["max_prompt_tokens"] <= 8192
        assert hashlib.sha256(tok.chat_template.encode()).hexdigest() == m["chat_template_sha"]
        cov_t = len({int(r["game_id"]) for r in
                     (json.loads(x) for x in gzip.open(path, "rt") if x.strip())
                     if r["label"] == "true"} & train_ids) / len(train_ids)
        print(f"[selfcheck] {args.tag}/{size}: OK — {m['n_items']} items ({n_t}T/{n_f}F), "
              f"TRUE game coverage {cov_t:.3f} (irrelevant under sampling=random), "
              f"origins={m['false_by_origin']}")
    for k, v in _saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="variant name; output dir <out-root>/<tag>/")
    ap.add_argument("--false-origin", default="",
                    help="comma list of origin_ckpt values to keep for gold-FALSE items "
                         "(e.g. 'base'); empty = all origins")
    ap.add_argument("--n-false", type=int, default=None,
                    help="target gold-FALSE item count (whole games are drawn until reached); "
                         "omit to keep every FALSE item surviving --false-origin")
    ap.add_argument("--n-false-4B", type=int, default=None, help="per-size override of --n-false")
    ap.add_argument("--n-false-8B", type=int, default=None, help="per-size override of --n-false")
    ap.add_argument("--nest-in", default="",
                    help="tag of a larger variant whose FALSE games this one must be a subset of "
                         "(makes the dose levels a nested ladder)")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--train-parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    ap.add_argument("--selfcheck", action="store_true", help="validate after writing")
    ap.add_argument("--selfcheck-only", action="store_true", help="validate an existing variant")
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out_root)
    if args.selfcheck_only:
        selfcheck(args, out_root)
        return
    parent_meta = json.loads((src_root / "_meta.json").read_text())
    write_variant(args, src_root, out_root, parent_meta)
    if args.selfcheck and not args.dry_run:
        selfcheck(args, out_root)


if __name__ == "__main__":
    main()
