#!/usr/bin/env python3
"""Build the endorsement-grid probe set: exactly N fabrications per (policy size, origin checkpoint).

Per (size in {4B, 8B}) x (ckpt in {base, gs50, gs100}) cell, the pool is that checkpoint's
fabrication items in the primary tables under --src-items (incontext__/contexts__{size}__{ckpt},
from build_probe_items.py), plus the extra draw candidates__{size}__{ckpt}.jsonl under --raw-dir
when present (from --emit-candidates). Duplicates (same game_id and normalised text) keep the first
occurrence (primary first, then the extra draw, each in item_id order). No probe answer is read.

Draw: the pool sorted by item_id, then random.Random(f"{seed}|{size}|{ckpt}").sample(pool, N).
Fails if a cell has < N candidates; warns if the draw is near-exhaustive. Controls are not sampled:
each drawn context brings its control_real and control_distractor, and the context table holds
exactly those contexts.

Ids: primary rows keep item_id, with context_id = {size}_{ckpt}_g{g}_r{r}; extra-draw rows carry the
draw tag in both ids ({size}_{ckpt}_{draw}_g{g}_r{r}, item_id + "_{draw}") and a `draw` field.
Writes union__{size}.jsonl, contexts_union__{size}.jsonl.gz, union_smoke__{size}.jsonl and
_union_summary.json (per-size union_build_id = sha256 of the item table, counts, draw record).

  python evaluation/source_of_fabrication/build_item_set.py --emit-candidates \
      --raw-dir $SOF_ROOT/probe_items/d2 --size 8B --ckpt base --draw d2
  python evaluation/source_of_fabrication/build_item_set.py --sizes 4B 8B --selftest \
      --src-items $SOF_ROOT/probe_items/primary --out-dir $SOF_ROOT/probe_items_union
"""
import argparse
import glob
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from rl.monitors import _norm  # noqa: E402

SIZES = ["4B", "8B"]
CKPTS = ["base", "gs50", "gs100"]
CKPT_STEP = {"base": 0, "gs50": 50, "gs100": 100}
KINDS = ("fabrication", "control_real", "control_distractor")
# Warn when the pool is below this multiple of N (the draw is near-exhaustive).
THIN_POOL_FACTOR = 1.25

DRAW_RULE = ("per cell: the pooled fabrication candidates sorted by item_id, then N drawn uniformly at random "
             "with random.Random(f'{seed}|{size}|{ckpt}').sample")


# io
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def read_gz(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_gz(path, rows):
    with gzip.GzipFile(path, "wb", mtime=0) as gz:      # mtime=0 -> reproducible bytes / build id
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode())


# Namespacing
def _check_raw(it, size, ck):
    if it.get("size") != size or it.get("ckpt") != ck:
        raise SystemExit(f"[union] {size}/{ck}: item {it.get('item_id')} has size={it.get('size')} ckpt={it.get('ckpt')}")
    if it.get("kind") not in KINDS:
        raise SystemExit(f"[union] {size}/{ck}: bad kind {it.get('kind')!r}")
    if it.get("round") not in (1, 2, 3):
        raise SystemExit(f"[union] {size}/{ck}: bad round {it.get('round')!r}")
    prefix = f"{size}_{ck}_g{it['game_id']}_r{it['round']}_"
    if not str(it["item_id"]).startswith(prefix):
        raise SystemExit(f"[union] item_id {it['item_id']!r} does not start with {prefix!r}")


def ns_item(it, size, ck, draw=None):
    """Primary transform (draw=None) or the extra-draw transform (context_id + item_id carry the draw)."""
    _check_raw(it, size, ck)
    new = dict(it)
    new["context_id_orig"] = it["context_id"]
    tag = f"{size}_{ck}" + (f"_{draw}" if draw else "")
    new["context_id"] = f"{tag}_g{it['game_id']}_r{it['round']}"
    new["origin_ckpt"] = ck
    new["origin_step"] = CKPT_STEP[ck]
    new["design"] = "incontext_union"
    if draw:
        new["item_id_orig"] = it["item_id"]
        new["item_id"] = f"{it['item_id']}_{draw}"
        new["draw"] = draw
    assert new["origin_ckpt"] == new["ckpt"] and new["origin_step"] == new["step"], new["item_id"]
    return new


def ns_ctx(c, size, ck, draw=None):
    nc = dict(c)
    nc["context_id_orig"] = c["context_id"]
    tag = f"{size}_{ck}" + (f"_{draw}" if draw else "")
    nc["context_id"] = f"{tag}_g{c['game_id']}_r{c['round']}"
    nc["origin_ckpt"] = ck
    nc["origin_step"] = CKPT_STEP[ck]
    if draw:
        nc["draw"] = draw
    return nc


def emit_candidates(raw_dir, size, ck, draw):
    raw_dir = Path(raw_dir)
    items = read_jsonl(raw_dir / f"incontext__{size}__{ck}.jsonl")
    ctxs = read_gz(raw_dir / f"contexts__{size}__{ck}.jsonl.gz")
    cand = [ns_item(it, size, ck, draw) for it in items]
    cctx = [ns_ctx(c, size, ck, draw) for c in ctxs]
    cand.sort(key=lambda x: (x["game_id"], x["round"], x["item_id"]))
    cctx.sort(key=lambda c: c["context_id"])
    ids = [x["item_id"] for x in cand]
    if len(set(ids)) != len(ids):
        raise SystemExit("[cand] duplicate item_ids after namespacing")
    missing = {x["context_id"] for x in cand} - {c["context_id"] for c in cctx}
    if missing:
        raise SystemExit(f"[cand] {len(missing)} item context_ids unresolvable e.g. {sorted(missing)[:3]}")
    ip, cp = raw_dir / f"candidates__{size}__{ck}.jsonl", raw_dir / f"candidates_contexts__{size}__{ck}.jsonl.gz"
    write_jsonl(ip, cand)
    write_gz(cp, cctx)
    n_fab = sum(x["kind"] == "fabrication" for x in cand)
    summ = {"size": size, "ckpt": ck, "draw": draw, "n_items": len(cand), "n_fab": n_fab,
            "n_ctrl": len(cand) - n_fab, "n_contexts": len(cctx),
            "n_games": len({x["game_id"] for x in cand}),
            "by_round_fab": dict(Counter(x["round"] for x in cand if x["kind"] == "fabrication")),
            "sha256": {ip.name: _sha256(ip), cp.name: _sha256(cp),
                       f"incontext__{size}__{ck}.jsonl": _sha256(raw_dir / f"incontext__{size}__{ck}.jsonl"),
                       f"contexts__{size}__{ck}.jsonl.gz": _sha256(raw_dir / f"contexts__{size}__{ck}.jsonl.gz")}}
    json.dump(summ, open(raw_dir / f"_candidates_summary__{size}__{ck}.json", "w"), indent=2)
    print(f"[cand] {size}/{ck}/{draw}: {n_fab} fabrication + {len(cand) - n_fab} control candidates over "
          f"{summ['n_games']} games, {len(cctx)} contexts -> {ip.name}")
    return summ


# The draw
def _fab_key(it):
    return (it["game_id"], _norm(it["statement"]))


def dedup_fabrications(rows):
    """Keep the first row per (game_id, normalised statement) in the given order -> (kept, n_dropped)."""
    seen, kept = set(), []
    for it in rows:
        k = _fab_key(it)
        if k in seen:
            continue
        seen.add(k)
        kept.append(it)
    return kept, len(rows) - len(kept)


def draw_cell(pool, n, rng):
    """N fabrication items from the cell's pool, sampled uniformly at random (see DRAW_RULE)."""
    if len(pool) < n:
        raise SystemExit(f"[union] only {len(pool)} fabrication candidates for N={n}")
    return rng.sample(sorted(pool, key=lambda it: it["item_id"]), n)


def extra_draw_cells(raw_dir, sizes):
    """{(size, ckpt)} whose extra-draw candidates file exists under `raw_dir`."""
    if not raw_dir:
        return set()
    raw_dir = Path(raw_dir)
    return {(size, ck) for size in sizes for ck in CKPTS
            if (raw_dir / f"candidates__{size}__{ck}.jsonl").exists()}


# Per-size build
def load_primary(src, size, ck):
    items, ctxs = Path(src) / f"incontext__{size}__{ck}.jsonl", Path(src) / f"contexts__{size}__{ck}.jsonl.gz"
    for f in (items, ctxs):
        if not f.is_file():
            raise SystemExit(f"[union] missing {f}: the item set needs every generation cell of a size "
                             f"({', '.join(CKPTS)}); this one is written by "
                             f"`sbatch --export=ALL,SIZE={size},CKPT={ck} "
                             f"evaluation/source_of_fabrication/generate_items.slurm`")
    return read_jsonl(items), read_gz(ctxs)


def load_extra(raw_dir, size, ck):
    raw_dir = Path(raw_dir)
    cands = read_jsonl(raw_dir / f"candidates__{size}__{ck}.jsonl")
    cctx = read_gz(raw_dir / f"candidates_contexts__{size}__{ck}.jsonl.gz")
    for c in cands:
        if not c.get("draw") or c.get("origin_ckpt") != ck or c.get("size") != size:
            raise SystemExit(f"[union] {size}/{ck}: candidate {c.get('item_id')} is not a namespaced extra-draw row")
    return cands, cctx


def val_ids_from_parquet(path):
    """Held-out game ids of the split parquet, or None if unreadable (then unchecked)."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return {int(e["index"]) for e in df["extra_info"]}
    except Exception as e:  # noqa: BLE001
        print(f"[union] WARN: could not read {path} ({e}); extra-draw games are not checked against the val split")
        return None


def _game_sig(it):
    return (it.get("background", ""), it.get("prior_belief", ""), tuple(it.get("evidence_descs") or []))


def build_size(size, args, extra_cells):
    val_ids = val_ids_from_parquet(args.val_parquet)
    items_out, ctx_out = [], {}
    game_content, fab_keys, draw = {}, {}, {}
    n_target = args.n_per_cell

    for ck in CKPTS:
        raw_items, raw_ctxs = load_primary(args.src_items, size, ck)
        items = [ns_item(it, size, ck) for it in raw_items]
        ctxs = {c["context_id"]: c for c in (ns_ctx(c, size, ck) for c in raw_ctxs)}
        n_src = sum(it["kind"] == "fabrication" for it in items)
        by_source = {"primary": n_src}
        if (size, ck) in extra_cells:
            cands, cctx = load_extra(args.raw_dir, size, ck)
            if val_ids is not None:
                bad = {c["game_id"] for c in cands} - val_ids
                if bad:
                    raise SystemExit(f"[union] {size}/{ck}: extra-draw games outside the val-100 split: {sorted(bad)[:5]}")
            for c in cands:
                by_source[c["draw"]] = by_source.get(c["draw"], 0) + (c["kind"] == "fabrication")
            items = items + sorted(cands, key=lambda x: x["item_id"])
            for c in cctx:
                if c["context_id"] in ctxs:
                    raise SystemExit(f"[union] {size}/{ck}: extra-draw context {c['context_id']} collides with a primary one")
                ctxs[c["context_id"]] = c
        for it in items:
            if game_content.setdefault(it["game_id"], _game_sig(it)) != _game_sig(it):
                raise SystemExit(f"[union] {size}: game {it['game_id']} case material differs across origins or draws")
        fab_all = [it for it in items if it["kind"] == "fabrication"]
        fab, n_dropped = dedup_fabrications(fab_all)

        rng = random.Random(f"{args.seed}|{size}|{ck}")
        try:
            sel = draw_cell(fab, n_target, rng)
        except SystemExit as e:
            raise SystemExit(f"{e} ({size}/{ck})") from None
        if len(fab) < THIN_POOL_FACTOR * n_target:
            print(f"[union] WARN {size}/{ck}: {len(fab)} candidates for N={n_target} — the draw is "
                  f"near-exhaustive, so this cell is barely a random sample of what the checkpoint "
                  f"fabricated")
        sel_ctx_ids = {it["context_id"] for it in sel}
        ctrls = [it for it in items if it["kind"] != "fabrication" and it["context_id"] in sel_ctx_ids]
        items_out.extend(sel + ctrls)
        for cid in sel_ctx_ids:
            ctx_out[cid] = ctxs[cid]
        fab_keys[ck] = {_fab_key(it) for it in sel}

        cell = {"mode": "random_draw", "draw_rule": DRAW_RULE,
                "n_candidates_fab_raw": len(fab_all), "n_dedup_dropped": n_dropped,
                "n_candidates_fab": len(fab), "by_source": by_source,
                "n_candidate_games": len({it["game_id"] for it in fab}),
                "by_round_candidates": dict(Counter(it["round"] for it in fab)),
                "n_selected": len(sel), "n_games_selected": len({it["game_id"] for it in sel}),
                "max_per_game": max(Counter(it["game_id"] for it in sel).values()),
                "by_round_selected": dict(Counter(it["round"] for it in sel)),
                "by_draw": dict(Counter(it.get("draw") or "primary" for it in sel)),
                "n_controls": len(ctrls), "n_contexts": len(sel_ctx_ids)}
        draw[ck] = cell
        print(f"[union] {size}/{ck}: {len(sel)} fabrications drawn from {len(fab)} candidates "
              f"({', '.join(f'{k} {v}' for k, v in by_source.items())}; {n_dropped} duplicate dropped) over "
              f"{cell['n_games_selected']} games (max {cell['max_per_game']}/game), by draw {cell['by_draw']}; "
              f"{len(ctrls)} controls, {len(sel_ctx_ids)} contexts")

    # Integrity checks (all fatal)
    for a, b in (("base", "gs50"), ("base", "gs100"), ("gs50", "gs100")):
        shared = fab_keys[a] & fab_keys[b]
        if shared:
            raise SystemExit(f"[union] {size}: {len(shared)} fabrications shared between {a} and {b}")
    items_out.sort(key=lambda x: (x["origin_step"], x["game_id"], x["round"], x["item_id"]))
    ids = [x["item_id"] for x in items_out]
    if len(set(ids)) != len(ids):
        dup = [k for k, v in Counter(ids).items() if v > 1][:5]
        raise SystemExit(f"[union] {size}: duplicate item_ids e.g. {dup}")
    ctx_list = sorted(ctx_out.values(), key=lambda c: c["context_id"])
    missing = {x["context_id"] for x in items_out} - set(ctx_out)
    if missing:
        raise SystemExit(f"[union] {size}: {len(missing)} item context_ids unresolvable e.g. {sorted(missing)[:3]}")
    for x in items_out:
        if x.get("origin_ckpt") != x.get("ckpt") or x.get("origin_step") != x.get("step"):
            raise SystemExit(f"[union] {size}: inconsistent origin label on {x['item_id']}")
    per_origin = {ck: {"n_items": sum(x["origin_ckpt"] == ck for x in items_out),
                       "n_fab": sum(x["origin_ckpt"] == ck and x["kind"] == "fabrication" for x in items_out),
                       "n_contexts": sum(c["origin_ckpt"] == ck for c in ctx_list)} for ck in CKPTS}
    for ck in CKPTS:
        if per_origin[ck]["n_fab"] != n_target:
            raise SystemExit(f"[union] {size}/{ck}: n_fab {per_origin[ck]['n_fab']} != {n_target}")
    n_fab = sum(x["kind"] == "fabrication" for x in items_out)
    summary = {
        "size": size, "n_items": len(items_out), "n_fab": n_fab, "n_ctrl": len(items_out) - n_fab,
        "n_contexts": len(ctx_list), "per_origin": per_origin,
        "n_games": len({x["game_id"] for x in items_out}),
        "by_kind": dict(Counter(x["kind"] for x in items_out)),
        "by_round_fab": dict(Counter(x["round"] for x in items_out if x["kind"] == "fabrication")),
        "draw": draw,
    }
    return items_out, ctx_list, summary


def smoke_subset(items, seed=0):
    """~25-item smoke set covering the design: per origin 2 fabrications (extra-draw ones first where
    they exist), one per round, one control of each kind."""
    rng = random.Random(seed)
    picked, seen = [], set()

    def take(pool, n):
        for it in rng.sample(pool, min(n, len(pool))):
            if it["item_id"] not in seen:
                seen.add(it["item_id"])
                picked.append(it)

    for ck in CKPTS:
        fab = [x for x in items if x["origin_ckpt"] == ck and x["kind"] == "fabrication"]
        d2 = [x for x in fab if x.get("draw")]
        if d2:
            take(d2, 2)
        take(fab, 2)
        for r_ in (1, 2, 3):
            take([x for x in fab if x["round"] == r_], 1)
        take([x for x in items if x["origin_ckpt"] == ck and x["kind"] == "control_real"], 1)
        take([x for x in items if x["origin_ckpt"] == ck and x["kind"] == "control_distractor"], 1)
    picked.sort(key=lambda x: (x["origin_step"], x["game_id"], x["round"], x["item_id"]))
    return picked


def selftest(size, union_items, union_ctxs, args):
    """Check composed messages from the union tables are byte-equal to those from the source tables,
    for every item and (mode, arm); catches splicing a claim into the wrong transcript."""
    from evaluation.source_of_fabrication.probe_prompts import compose
    uctx = {c["context_id"]: c for c in union_ctxs}
    ref = {}
    for ck in CKPTS:
        ri, rc = load_primary(args.src_items, size, ck)
        ref[(ck, None)] = ({x["item_id"]: x for x in ri}, {c["context_id"]: c for c in rc})
    draws = {(x["origin_ckpt"], x["draw"]) for x in union_items if x.get("draw")}
    if draws and not args.raw_dir:
        raise SystemExit(f"[selftest] {size}: the item set carries extra-draw rows {sorted(draws)} "
                         f"but --raw-dir was not given, so they cannot be re-composed from source")
    raw_dir = Path(args.raw_dir) if draws else None
    for ck, d in draws:
        ri = read_jsonl(raw_dir / f"incontext__{size}__{ck}.jsonl")
        rc = read_gz(raw_dir / f"contexts__{size}__{ck}.jsonl.gz")
        ref[(ck, d)] = ({x["item_id"]: x for x in ri}, {c["context_id"]: c for c in rc})
    n = 0
    for it in union_items:
        ritems, rctx = ref[(it["origin_ckpt"], it.get("draw"))]
        r = ritems[it.get("item_id_orig", it["item_id"])]
        for mode, arm in (("out", "honest"), ("in", "honest"), ("in", "role")):
            if compose(it, uctx, mode, arm, "pre") != compose(r, rctx, mode, arm, "pre"):
                raise SystemExit(f"[selftest] {size} {it['item_id']} [{mode}/{arm}]: composed messages DIFFER")
            n += 1
    print(f"[selftest] {size}: {n} composed-message byte-equality checks passed")
    return n


def gather_fingerprints(paths):
    fps = {}
    for pat in paths:
        for f in sorted(glob.glob(str(pat))):
            d = json.load(open(f))
            lab = d.get("label") or d.get("basename")
            rec = {"fingerprint": d["fingerprint"], "basename": d["basename"], "path": d["path"],
                   "n_files": d["n_files"], "total_bytes": d["total_bytes"], "source_file": f}
            if lab in fps and fps[lab]["fingerprint"] != rec["fingerprint"]:
                raise SystemExit(f"[fp] {lab}: fingerprint in {f} differs from {fps[lab]['source_file']}")
            fps.setdefault(lab, rec)
    return fps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--emit-candidates", action="store_true",
                    help="namespace a draw's raw probe items into candidates under --raw-dir")
    ap.add_argument("--raw-dir", default=None,
                    help="extra sampling draws: a cell whose candidates__{size}__{ckpt}.jsonl is "
                         "here pools that draw too; written by --emit-candidates")
    ap.add_argument("--rollouts-dir", default=None,
                    help="dir holding the stage-A _ckpt__*.json checkpoint fingerprints to record")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--draw", default="d2")
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--src-items", default=None,
                    help="dir of the primary incontext__/contexts__ tables (build_probe_items.py)")
    ap.add_argument("--out-dir", default=None, help="dir the item set is written to")
    ap.add_argument("--profile", default="stubborn",
                    help="receiver profile whose GRPO split the --val-parquet default names")
    ap.add_argument("--val-parquet", default=None,
                    help="held-out split parquet the extra-draw games are checked against "
                         "(default datasets/old_bailey/_generated/rl/<profile>/rl_validation.parquet)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.val_parquet is None:
        args.val_parquet = (_REPO / "datasets/old_bailey/_generated/rl" / args.profile
                            / "rl_validation.parquet")

    if args.emit_candidates:
        if not args.raw_dir:
            ap.error("--raw-dir is required with --emit-candidates")
        emit_candidates(args.raw_dir, args.size, args.ckpt, args.draw)
        return
    for req in ("src_items", "out_dir"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required")
    extra_cells = extra_draw_cells(args.raw_dir, args.sizes)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Merge: a per-size build must keep the other size's entry, or the classify completeness gate
    # skips that size and probe_cell.slurm cannot resume it.
    spath = out / "_union_summary.json"
    prev = json.load(open(spath)) if spath.exists() else {}
    input_dirs = {"primary"} | ({Path(args.raw_dir).name} if args.raw_dir else set())
    summary = {"design": "N fabrications per (size, origin checkpoint), drawn at random from the pooled candidates "
                         "(the primary tables, plus any extra draw on disk for that cell); controls = "
                         "the two of each selected context",
               "n_per_cell": args.n_per_cell, "seed": args.seed, "draw_rule": DRAW_RULE,
               "extra_candidate_cells": [f"{s}/{c}" for s, c in sorted(extra_cells)],
               "sizes": dict(prev.get("sizes", {})),
               "inputs": {k: v for k, v in prev.get("inputs", {}).items() if k.split("/", 1)[0] in input_dirs}}
    for size in args.sizes:
        items, contexts, s = build_size(size, args, extra_cells)
        items_path = out / f"union__{size}.jsonl"
        ctxs_path = out / f"contexts_union__{size}.jsonl.gz"
        smoke_path = out / f"union_smoke__{size}.jsonl"
        write_jsonl(items_path, items)
        write_gz(ctxs_path, contexts)
        smoke = smoke_subset(items)
        write_jsonl(smoke_path, smoke)
        if args.selftest:
            s["selftest_checks"] = selftest(size, items, contexts, args)
        s["union_build_id"] = _sha256(items_path)
        s["sha256"] = {items_path.name: s["union_build_id"], ctxs_path.name: _sha256(ctxs_path),
                       smoke_path.name: _sha256(smoke_path)}
        s["n_smoke"] = len(smoke)
        summary["sizes"][size] = s
        for ck in CKPTS:
            for stem in (f"incontext__{size}__{ck}.jsonl", f"contexts__{size}__{ck}.jsonl.gz"):
                summary["inputs"][f"primary/{stem}"] = _sha256(Path(args.src_items) / stem)
            if (size, ck) in extra_cells:
                for stem in (f"candidates__{size}__{ck}.jsonl", f"candidates_contexts__{size}__{ck}.jsonl.gz",
                             f"incontext__{size}__{ck}.jsonl", f"contexts__{size}__{ck}.jsonl.gz"):
                    summary["inputs"][f"{Path(args.raw_dir).name}/{stem}"] = _sha256(Path(args.raw_dir) / stem)
        print(f"[union] {size}: {s['n_items']} items ({s['n_fab']} fab / {s['n_ctrl']} ctrl), "
              f"{s['n_contexts']} contexts, {s['n_games']} games, smoke={len(smoke)} -> {items_path.name}")
    summary["ckpt_fingerprints"] = (gather_fingerprints([Path(args.rollouts_dir) / "_ckpt__*.json"])
                                    if args.rollouts_dir else {})
    json.dump(summary, open(out / "_union_summary.json", "w"), indent=2)
    print(f"[union] summary -> {out / '_union_summary.json'}")


if __name__ == "__main__":
    main()
