#!/usr/bin/env python3
"""Build an SFT parquet from an aux-CE probe sidecar, the offline half of the sequential study.

AUX_CE=1 (rl/aux_ce.py) adds CE on the gold `true`/`false` token of in-role deception-probe items
to every GRPO step. The sequential study trains the same items on the same objective as a separate
SFT stage before GRPO; this script writes the parquet scripts/sft_train_sender.slurm consumes.

    .venv/bin/python datasets/old_bailey/aux_loss/build_auxce_sft_dataset.py \\
        --sidecar datasets/old_bailey/_generated/aux_ce_probe/variants/base_origin/4B/sidecar.jsonl.gz \\
        --meta    datasets/old_bailey/_generated/aux_ce_probe/variants/base_origin/_meta.json \\
        --size 4B \\
        --tokenizer experiments/results/sft/sft_qwen3-4B_strategies_stubborn_ep2_misrepv2/hf_final \\
        --out-dir datasets/old_bailey/_generated/sft/auxce_base_origin_4B

Each output row is the sidecar's probe chat (ending on the probe user turn) plus the gold answer as
a final assistant turn, verbatim from rl/aux_ce.RESP_TEXT (no `<prob>` line, matching aux_ce).
Interior assistant turns get `loss_mask: 0`. The in-GRPO term scores only the label token, so an
SFT dataset class that does not narrow the final turn to it trains ~8x more tokens per item.

Build-time guards: sidecar sha256 and per-size counts match _meta.json; the tokenizer's
chat_template sha matches the sidecar's; every row fits --max-len (trainer uses
data.truncation=error); roles are system,(user,assistant)*,user,assistant.

Val is split by game. sft_val.parquet measures probe accuracy; it does not select a checkpoint
(only the last step is saved). --val-games 0 disables the holdout.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(REPO / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_AUX = _load_by_path("_pg_aux_ce", "rl/aux_ce.py")
RESP_TEXT = _AUX.RESP_TEXT
LABELS_ALL = _AUX.LABELS_ALL
# The trainer renders through this patch (empty-think stub on Qwen3-8B, no-op on
# Qwen3-4B-Instruct-2507), so the length guard must use it too.
patch_qwen3_thinking_template = _load_by_path(
    "_pg_sft_dataset", "sft/sft_dataset.py").patch_qwen3_thinking_template


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chat_template_sha(tokenizer) -> str:
    """sha256 of the rendered chat template — the digest _meta.json records per size."""
    tpl = tokenizer.chat_template or ""
    return hashlib.sha256(tpl.encode()).hexdigest()


def build_rows(sidecar_rows, source_file):
    """sidecar rows -> SFT rows (messages + provenance). Raises on any shape violation."""
    out = []
    for r in sidecar_rows:
        label = r["label"]
        if label not in LABELS_ALL:
            raise ValueError(f"{r['item_id']}: label {label!r} not in {LABELS_ALL}")
        msgs = [dict(m) for m in r["messages"]]
        roles = [m["role"] for m in msgs]
        # verl's MultiTurnSFTDataset checks this only in __getitem__, i.e. mid-epoch.
        if roles[0] != "system" or roles[-1] != "user" or len(roles) % 2 != 0:
            raise ValueError(f"{r['item_id']}: probe chat must be system,(user,assistant)*,user; "
                             f"got {roles}")
        if roles[1::2] != ["user"] * (len(roles) // 2):
            raise ValueError(f"{r['item_id']}: odd positions must all be 'user', got {roles}")
        if roles[2::2] != ["assistant"] * (len(roles) // 2 - 1):
            raise ValueError(f"{r['item_id']}: interior even positions must all be 'assistant', "
                             f"got {roles}")
        for m in msgs:
            # interior sender arguments are context, not targets
            m["loss_mask"] = 0 if m["role"] == "assistant" else None
        msgs.append({"role": "assistant", "content": RESP_TEXT[label], "loss_mask": None})

        out.append({
            "messages": msgs,
            "enable_thinking": False,
            "item_id": r["item_id"],
            "game_id": int(r["game_id"]),
            "kind": r["kind"],
            "pair_id": r.get("pair_id", ""),
            "label": label,
            "round": int(r["round"]),
            "origin_ckpt": r.get("origin_ckpt", ""),
            "context_id": r.get("context_id", ""),
            "statement": r.get("statement", ""),
            "source_file": str(source_file),
        })
    return out


def _render_ids(tokenizer, msgs):
    """Token ids for a rendered chat.

    transformers>=5 returns a BatchEncoding whose len() is its key count (same trap as
    rl/patch_verl.sh (12)), so unwrap input_ids and the batch dim.
    """
    out = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not (isinstance(ids, list) and (not ids or isinstance(ids[0], int))):
        raise TypeError(f"could not reduce apply_chat_template output to a token list: {type(out)}")
    return ids


def check_lengths(rows, tokenizer, max_len):
    """Render every row; return (max_tokens, histogram). Raises if any row exceeds max_len."""
    lens = []
    for r in rows:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        n = len(_render_ids(tokenizer, msgs))
        if n > max_len:
            raise ValueError(
                f"{r['item_id']} renders to {n} tokens > --max-len {max_len}; the trainer runs "
                "data.truncation=error and would die mid-epoch")
        lens.append(n)
    return max(lens), lens


A_KINDS = frozenset(("fabrication", "claim_real"))
B_KINDS = frozenset(("altered_real", "paraphrase_real"))


def _trim_train_by_pairs(train_rows, n_a, n_b, seed):
    """Trim the train split to exactly n_a block-A and n_b block-B pairs.

    Whole pairs only: pair halves are length/style-matched, so dropping one half puts the length
    confound back into the label; pairs split by the val boundary are dropped too. Applied to train,
    not the total, because block-A pairs can span two games and val is split by game.
    """
    by_pair = {}
    for r in train_rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    if "" in by_pair:
        raise SystemExit("ERROR: --train-pairs-* needs pair_id on every row; this variant has none "
                         "(only rebalanced variants are paired)")
    split_by_val = [k for k, v in by_pair.items() if len(v) != 2]
    pools = {"A": [], "B": []}
    for k, v in by_pair.items():
        if len(v) != 2:
            continue
        kinds = {r["kind"] for r in v}
        if kinds == A_KINDS:
            pools["A"].append(k)
        elif kinds == B_KINDS:
            pools["B"].append(k)
        else:
            raise SystemExit(f"ERROR: pair {k} has kinds {sorted(kinds)}, not a block-A or -B pair")
    for blk, want in (("A", n_a), ("B", n_b)):
        if len(pools[blk]) < want:
            raise SystemExit(
                f"ERROR: train split has {len(pools[blk])} complete block-{blk} pairs but "
                f"{want} were requested. Lower the target or shrink the val holdout "
                f"({len(split_by_val)} pair(s) were split by the val boundary and dropped).")
    rng = random.Random(seed)
    keep, used_cross = set(), 0
    for blk, want in (("A", n_a), ("B", n_b)):
        # Prefer same-game pairs: a cross-game pair can leave a game single-class, so the case
        # predicts the label (the confound enforce_game_balance prevents). Only block A has
        # cross-game pairs.
        same_set = {k for k in pools[blk] if len({r["game_id"] for r in by_pair[k]}) == 1}
        same = sorted(same_set)
        cross = sorted(k for k in pools[blk] if k not in same_set)
        rng.shuffle(same)
        rng.shuffle(cross)
        take = same[:want]
        if len(take) < want:
            need = want - len(take)
            take = take + cross[:need]
            used_cross += min(need, len(cross))
        if len(take) < want:
            raise SystemExit(
                f"ERROR: only {len(take)} block-{blk} pairs available in train, need {want}")
        keep.update(take)
    out = [r for r in train_rows if r["pair_id"] in keep]
    # Cross-game pairs may still have left a game single-class.
    per_game_labels = {}
    for r in out:
        per_game_labels.setdefault(r["game_id"], set()).add(r["label"])
    bad = sorted(g for g, L in per_game_labels.items() if len(L) < 2)
    if bad:
        raise SystemExit(
            f"ERROR: {len(bad)} game(s) end up single-class after the trim (e.g. {bad[:5]}) -- the "
            f"case would predict the label. {used_cross} cross-game pair(s) were needed to reach "
            "the budget; lower --train-pairs-* or enlarge the pool.")
    print(f"[trim] all {len(per_game_labels)} train games carry both classes "
          f"({used_cross} cross-game pair(s) used)")
    got = Counter(r["kind"] for r in out)
    exp = {"fabrication": n_a, "claim_real": n_a, "altered_real": n_b, "paraphrase_real": n_b}
    if dict(got) != exp:
        raise SystemExit(f"ERROR: trim produced {dict(got)}, expected {exp}")
    print(f"[trim] train -> {len(out)} rows: {dict(sorted(got.items()))} "
          f"({n_a} A-pairs + {n_b} B-pairs); dropped {len(split_by_val)} val-split pair(s), "
          f"{len(pools['A']) - n_a} spare A, {len(pools['B']) - n_b} spare B")
    return out


def _val_games_by_item_target(rows, games, target, rng):
    """Whole games whose item counts sum to exactly `target`.

    Whole games keep a game's frozen context out of both splits. An exact item target gives equal
    train rows (so equal optimizer steps) across sizes, which a fixed game count does not (games
    hold 1-4 items; base_origin: 1294/1294 items but 1182/1199 train rows). Greedy over a seeded
    shuffle plus an exact-fit repair; raises instead of returning a near miss.
    """
    per_game = {}
    for r in rows:
        per_game[r["game_id"]] = per_game.get(r["game_id"], 0) + 1
    total = sum(per_game.values())
    if target >= total:
        raise SystemExit(f"ERROR: --val-items {target} >= total items {total}")
    order = list(games)
    rng.shuffle(order)
    chosen, got = set(), 0
    for g in order:
        n = per_game[g]
        if got + n <= target:
            chosen.add(g)
            got += n
        if got == target:
            return chosen
    # close the remainder with one unused game of exactly that size...
    need = target - got
    for g in order:
        if g not in chosen and per_game[g] == need:
            chosen.add(g)
            return chosen
    # ...or swap a chosen game for an unused one exactly `need` larger (a remainder of 1 is common;
    # the smallest game holds 2 items).
    for g_out in sorted(chosen):
        want = per_game[g_out] + need
        for g_in in order:
            if g_in not in chosen and per_game[g_in] == want:
                chosen.discard(g_out)
                chosen.add(g_in)
                return chosen
    raise SystemExit(
        f"ERROR: cannot hit --val-items {target} exactly with whole games (reached {got}; "
        f"need a spare game of exactly {need} items). Game sizes present: "
        f"{sorted(set(per_game.values()))}. Pick a reachable target or use --val-games.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecar", required=True, help="<variant>/<size>/sidecar.jsonl.gz")
    ap.add_argument("--meta", required=True, help="the variant's sibling _meta.json")
    ap.add_argument("--size", required=True, choices=("4B", "8B", "14B"))
    ap.add_argument("--tokenizer", required=True,
                    help="the SFT INIT checkpoint whose template must match the sidecar's")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=8192,
                    help="must equal the trainer's MAX_LEN (data.truncation=error tripwire)")
    ap.add_argument("--val-items", type=int, default=0,
                    help="hold out whole games until the val split holds EXACTLY this many items "
                         "(0 = use --val-games). Equalizes TRAIN ROWS across sizes when both are "
                         "built from equal-sized variants, which --val-games cannot do: games "
                         "carry 1-4 items each, so a fixed GAME count yields different item counts "
                         "per size. Still selects whole games, so no game straddles the split.")
    ap.add_argument("--val-games", type=int, default=30,
                    help="held-out GAMES whose every item goes to sft_val.parquet (0 = none)")
    ap.add_argument("--train-pairs-a", type=int, default=0,
                    help="trim the TRAIN split to exactly this many block-A pairs "
                         "(fabrication + claim_real). 0 = no trim.")
    ap.add_argument("--train-pairs-b", type=int, default=0,
                    help="trim the TRAIN split to exactly this many block-B pairs "
                         "(altered_real + paraphrase_real). 0 = no trim.")
    ap.add_argument("--pair-seed", type=int, default=0)
    ap.add_argument("--dump-train-sidecar", default=None,
                    help="write the KEPT TRAIN rows back out in sidecar format, so "
                         "check_statement_balance.py can gate exactly what is trained "
                         "on. Trimming can leave a game single-class and can move the surface "
                         "AUCs, so a trimmed set that has not been re-gated is not gated.")
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--skip-template-check", action="store_true",
                    help="ESCAPE HATCH: only for a deliberate cross-template build")
    a = ap.parse_args(argv)

    sidecar = Path(a.sidecar).resolve()
    meta_path = Path(a.meta).resolve()
    out_dir = Path(a.out_dir).resolve()
    meta = json.loads(meta_path.read_text())
    size_meta = meta["sizes"][a.size]

    # (G1) sidecar integrity
    got = sha256_file(sidecar)
    # Index, not .get(), so a missing reference fails instead of skipping the check.
    want = size_meta["sidecar_sha256"]
    if got != want:
        raise SystemExit(f"ERROR: sidecar sha256 {got} != _meta.json {want} — stale or edited file")

    rows_in = [json.loads(line) for line in gzip.open(sidecar, "rt") if line.strip()]
    if len(rows_in) != size_meta["n_items"]:
        raise SystemExit(f"ERROR: {len(rows_in)} rows != _meta n_items {size_meta['n_items']}")
    lab = Counter(r["label"] for r in rows_in)
    if lab["true"] != size_meta["n_true"] or lab["false"] != size_meta["n_false"]:
        raise SystemExit(f"ERROR: label counts {dict(lab)} != _meta "
                         f"{size_meta['n_true']}T/{size_meta['n_false']}F")
    if any(r.get("term") for r in rows_in):
        raise SystemExit("ERROR: multi-term sidecar; this builder handles the single fabrication "
                         "channel only (the channel the chain's sidecars carry)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # (G2) chat-template parity with the sidecar's token check
    got_tpl, want_tpl = chat_template_sha(tok), size_meta["chat_template_sha"]
    if got_tpl != want_tpl and not a.skip_template_check:
        raise SystemExit(
            f"ERROR: tokenizer chat_template sha {got_tpl} != sidecar's {want_tpl}.\n"
            f"       tokenizer={a.tokenizer}\n"
            "       The probe contexts were composed under a different template; training on them "
            "would silently mis-render the frozen mid-game context. Use the matching checkpoint, "
            "or --skip-template-check if the mismatch is deliberate.")

    rows = build_rows(rows_in, sidecar)
    # measure through the trainer's patched template
    max_tok, lens = check_lengths(rows, patch_qwen3_thinking_template(tok), a.max_len)

    # game-level val split (no game straddles the boundary)
    games = sorted({r["game_id"] for r in rows})
    rng = random.Random(a.val_seed)
    if a.val_items > 0:
        val_games = _val_games_by_item_target(rows, games, a.val_items, rng)
    elif a.val_games > 0:
        val_games = set(rng.sample(games, min(a.val_games, len(games))))
    else:
        val_games = set()
    train_rows = [r for r in rows if r["game_id"] not in val_games]
    val_rows = [r for r in rows if r["game_id"] in val_games]
    if a.train_pairs_a or a.train_pairs_b:
        train_rows = _trim_train_by_pairs(train_rows, a.train_pairs_a, a.train_pairs_b, a.pair_seed)
    random.Random(a.shuffle_seed).shuffle(train_rows)

    if not train_rows:
        raise SystemExit("ERROR: empty train split")

    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove any old val split first: a stale sft_val.parquet would overlap train, and held-out
    # probe accuracy is scored from it.
    (out_dir / "sft_val.parquet").unlink(missing_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "sft_train.parquet")
    if a.dump_train_sidecar:
        keep_ids = {r["item_id"] for r in train_rows}
        dpath = Path(a.dump_train_sidecar)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0 so a rebuild is byte-identical, as write_sidecar does upstream
        with gzip.GzipFile(dpath, "wb", mtime=0) as fh:
            for r in rows_in:
                if r["item_id"] in keep_ids:
                    fh.write((json.dumps(r, sort_keys=True) + "\n").encode())
        print(f"[dump] {len(keep_ids)} train rows -> {dpath} (gate this, not the parent)")
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(out_dir / "sft_val.parquet")

    stats = {
        "builder": "datasets/old_bailey/aux_loss/build_auxce_sft_dataset.py",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "sidecar": str(sidecar),
        "sidecar_sha256": got,
        "variant_tag": meta.get("variant", {}).get("tag"),
        "size": a.size,
        "tokenizer": a.tokenizer,
        "chat_template_sha": got_tpl,
        "resp_text": {lb: RESP_TEXT[lb] for lb in LABELS_ALL},
        "n_items": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_games": len(games),
        "val_games": sorted(val_games),
        "label_counts_all": dict(lab),
        "label_counts_train": dict(Counter(r["label"] for r in train_rows)),
        "label_counts_val": dict(Counter(r["label"] for r in val_rows)),
        "kind_counts": dict(Counter(r["kind"] for r in rows)),
        "max_render_tokens": max_tok,
        "mean_render_tokens": round(sum(lens) / len(lens), 1),
        "max_len_guard": a.max_len,
    }
    (out_dir / "sft_build_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[build] {len(train_rows)} train / {len(val_rows)} val rows -> {out_dir}")
    print(f"[build] labels train={stats['label_counts_train']} val={stats['label_counts_val']}")
    print(f"[build] render tokens: max={max_tok} mean={stats['mean_render_tokens']} "
          f"(guard {a.max_len})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
