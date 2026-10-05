"""Verify a --dump-train-sidecar output against its parent and write the variant _meta.json.

build_auxce_sft_dataset.py --dump-train-sidecar writes an SFT corpus's kept train rows as a
sidecar, so GRPO aux-CE can train on exactly the SFT rows. rl/aux_ce.load_state and the submit gate
in scripts/rl_train_sender.slurm need a sibling variants/<tag>/_meta.json; before writing it, this
script checks per size:

  1. parent integrity: parent sidecar sha256 matches the parent _meta.json
  2. provenance: each dumped row is content-equal to its parent row, in parent order (not
     byte-equal, since the dump uses sort_keys)
  3. shape: expected item and per-kind counts, 1:1 labels, every pair_id whole with a block-A or
     block-B kind set and one row per label
  4. SFT == aux: dumped item_ids equal sft_train.parquet's
  5. frozen val: sft_val.parquet is row-identical to the reference corpus's
  6. render: rows fit the token cap under the pinned tokenizer, whose chat_template sha must equal
     the parent's; the sha is copied, never recomputed (cf. rl/CLAUDE.md)
It also asserts that n_items and by_kind match across --sizes.

  .venv/bin/python datasets/old_bailey/aux_loss/make_train_sidecar_meta.py \\
      --tag kind_balanced --parent balanced \\
      --sft-dir-tpl datasets/old_bailey/_generated/sft/auxce_kind_balanced_{size} \\
      --ref-sft-dir-tpl datasets/old_bailey/_generated/sft/auxce_balanced_{size} \\
      --expect-items 1924 --expect-pairs-a 481 --expect-pairs-b 481
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    A_KINDS,
    B_KINDS,
    DEFAULT_VARIANTS,
    KIND_BLOCK,
    check_render,
    git_commit,
    load_sidecar,
    sha256_file,
)


def die(msg: str):
    raise SystemExit(f"ERROR: {msg}")


def load_parquet_rows(path: Path):
    import pandas as pd

    if not path.is_file():
        die(f"missing parquet: {path}")
    return pd.read_parquet(path).to_dict("records")


def canon_messages(messages):
    """Content-comparison key for a messages list (parquet round-trips lists as ndarrays)."""
    return json.dumps([{"role": m["role"], "content": m["content"]} for m in messages],
                      sort_keys=True)


def verify_size(size: str, args, parent_meta: dict) -> dict:
    pm = parent_meta["sizes"].get(size) or die(f"parent _meta.json has no size {size!r}")
    root = Path(args.root)
    parent_sidecar = root / args.parent / size / "sidecar.jsonl.gz"
    dumped_sidecar = root / args.tag / size / "sidecar.jsonl.gz"

    # (1) parent integrity
    parent_sha = sha256_file(parent_sidecar)
    if parent_sha != pm["sidecar_sha256"]:
        die(f"[{size}] parent sidecar sha {parent_sha[:16]} != recorded {pm['sidecar_sha256'][:16]}")

    parent_rows = load_sidecar(parent_sidecar)
    parent_by_id = {}
    for i, r in enumerate(parent_rows):
        if r["item_id"] in parent_by_id:
            die(f"[{size}] duplicate item_id in PARENT: {r['item_id']!r}")
        parent_by_id[r["item_id"]] = (i, r)

    rows = load_sidecar(dumped_sidecar)

    # (2) provenance: content-equal rows, parent order preserved
    last_idx = -1
    for r in rows:
        got = parent_by_id.get(r["item_id"]) or die(f"[{size}] {r['item_id']!r} not in parent")
        idx, pr = got
        if r != pr:
            die(f"[{size}] row {r['item_id']!r} differs from its parent row")
        if idx <= last_idx:
            die(f"[{size}] {r['item_id']!r} out of parent order (dump is not a subsequence)")
        last_idx = idx

    # (3) shape: counts, balance, whole pairs, per-block label equality
    n = len(rows)
    if n != args.expect_items:
        die(f"[{size}] {n} items, expected {args.expect_items}")
    labels = Counter(r["label"] for r in rows)
    if labels["true"] != labels["false"] or set(labels) != {"true", "false"}:
        die(f"[{size}] label imbalance: {dict(labels)}")
    kinds = Counter(r["kind"] for r in rows)
    expect_kinds = {"fabrication": args.expect_pairs_a, "claim_real": args.expect_pairs_a,
                    "altered_real": args.expect_pairs_b, "paraphrase_real": args.expect_pairs_b}
    if dict(kinds) != expect_kinds:
        die(f"[{size}] kind counts {dict(kinds)}, expected {expect_kinds}")
    by_pair: dict[str, list] = {}
    for r in rows:
        if not r.get("pair_id"):
            die(f"[{size}] {r['item_id']!r} has no pair_id")
        by_pair.setdefault(r["pair_id"], []).append(r)
    block_cnt: Counter = Counter()
    for k, v in by_pair.items():
        if len(v) != 2:
            die(f"[{size}] pair {k!r} has {len(v)} rows, not 2 (half-pair in the train set)")
        ks = {r["kind"] for r in v}
        if ks not in (A_KINDS, B_KINDS):
            die(f"[{size}] pair {k!r} has kinds {sorted(ks)} — not a block-A or block-B pair")
        if {r["label"] for r in v} != {"true", "false"}:
            die(f"[{size}] pair {k!r} is not one TRUE + one FALSE")
        for r in v:
            block_cnt[(KIND_BLOCK[r["kind"]], r["label"])] += 1
    for blk in ("a", "b"):
        t, f = block_cnt[(blk, "true")], block_cnt[(blk, "false")]
        if not (t and f and t == f):
            die(f"[{size}] block {blk!r} label-imbalanced: {t}T/{f}F")

    # (4) requirement 1: aux set == SFT train set
    sft_dir = Path(args.sft_dir_tpl.format(size=size))
    train_ids = {r["item_id"] for r in load_parquet_rows(sft_dir / "sft_train.parquet")}
    side_ids = {r["item_id"] for r in rows}
    if train_ids != side_ids:
        die(f"[{size}] sidecar items != sft_train.parquet items "
            f"(only-sidecar {len(side_ids - train_ids)}, only-parquet {len(train_ids - side_ids)})")

    # (5) requirement 3: frozen val, row-identical to the reference corpus
    ref_dir = Path(args.ref_sft_dir_tpl.format(size=size))
    new_val = {r["item_id"]: r for r in load_parquet_rows(sft_dir / "sft_val.parquet")}
    ref_val = {r["item_id"]: r for r in load_parquet_rows(ref_dir / "sft_val.parquet")}
    if set(new_val) != set(ref_val):
        die(f"[{size}] val item_id sets differ from {ref_dir} "
            f"(new-only {len(set(new_val) - set(ref_val))}, ref-only {len(set(ref_val) - set(new_val))})")
    for iid, nr in new_val.items():
        rr = ref_val[iid]
        if nr["label"] != rr["label"] or canon_messages(nr["messages"]) != canon_messages(rr["messages"]):
            die(f"[{size}] val row {iid!r} differs from the reference corpus")
    if train_ids & set(new_val):
        die(f"[{size}] train/val overlap: {len(train_ids & set(new_val))} items")

    # (6) render under the pinned tokenizer; sha verified against the parent's, then COPIED
    cap = pm.get("max_prompt_tokens_cap", args.cap)
    max_tok = check_render(rows, size, cap, pm["chat_template_sha"])

    games = {int(r["game_id"]) for r in rows}
    games_t = {int(r["game_id"]) for r in rows if r["label"] == "true"}
    games_f = {int(r["game_id"]) for r in rows if r["label"] == "false"}
    print(f"[{size}] OK: {n} items ({labels['true']}T/{labels['false']}F), "
          f"kinds {dict(sorted(kinds.items()))}, {len(by_pair)} whole pairs, "
          f"{len(games)} games, max_prompt_tokens={max_tok}, sidecar==sft_train, val frozen")

    return {
        "n_items": n,
        "n_true": labels["true"],
        "n_false": labels["false"],
        "by_kind": dict(sorted(kinds.items())),
        "n_games": len(games),
        "n_games_with_true": len(games_t),
        "n_games_with_false": len(games_f),
        "tokenizer_path": pm["tokenizer_path"],
        "chat_template_sha": pm["chat_template_sha"],
        "max_prompt_tokens": max_tok,
        "max_prompt_tokens_cap": cap,
        "parent_sidecar_sha256": parent_sha,
        "parent_n_items": len(parent_rows),
        "sidecar_sha256": sha256_file(dumped_sidecar),
        "sft_dir": str(sft_dir),
        "n_val_frozen": len(new_val),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="the dumped variant, e.g. kind_balanced")
    ap.add_argument("--parent", required=True, help="the parent variant, e.g. balanced")
    ap.add_argument("--root", default=str(DEFAULT_VARIANTS))
    ap.add_argument("--sizes", nargs="+", default=["4B", "8B"])
    ap.add_argument("--sft-dir-tpl", required=True,
                    help="new corpus dir template, '{size}' substituted")
    ap.add_argument("--ref-sft-dir-tpl", required=True,
                    help="reference corpus whose sft_val.parquet must be reproduced exactly")
    ap.add_argument("--expect-items", type=int, required=True)
    ap.add_argument("--expect-pairs-a", type=int, required=True)
    ap.add_argument("--expect-pairs-b", type=int, required=True)
    ap.add_argument("--cap", type=int, default=8192,
                    help="fallback token cap if the parent meta lacks max_prompt_tokens_cap")
    args = ap.parse_args(argv)

    parent_meta_path = Path(args.root) / args.parent / "_meta.json"
    parent_meta = json.loads(parent_meta_path.read_text())

    sizes = {}
    for size in args.sizes:
        sizes[size] = verify_size(size, args, parent_meta)

    ref = sizes[args.sizes[0]]
    for size in args.sizes[1:]:
        m = sizes[size]
        if (m["n_items"], m["by_kind"]) != (ref["n_items"], ref["by_kind"]):
            die(f"sizes not equalized: {args.sizes[0]}={ref['n_items']}/{ref['by_kind']} "
                f"vs {size}={m['n_items']}/{m['by_kind']}")

    meta = {
        "schema_version": parent_meta.get("schema_version", 1),
        "arm": parent_meta.get("arm"),
        "split_file": parent_meta.get("split_file"),
        "git_commit": git_commit(_REPO),
        "variant": {
            "tag": args.tag,
            "parent": args.parent,
            "builder": ("datasets/old_bailey/aux_loss/build_auxce_sft_dataset.py "
                        "--train-pairs-a/--train-pairs-b --dump-train-sidecar; meta+proofs by "
                        "datasets/old_bailey/aux_loss/make_train_sidecar_meta.py"),
            "argv": sys.argv[1:],
            "design": ("train-only, pair-complete, per-kind size-equalized subset of the parent; "
                       "the frozen 30-game val holdout is excluded, so this sidecar holds exactly "
                       "the rows the matching SFT corpus trains on"),
            "parent_meta": str(parent_meta_path),
        },
        "sizes": sizes,
    }
    out = Path(args.root) / args.tag / "_meta.json"
    out.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[meta] wrote {out}")


if __name__ == "__main__":
    main()
