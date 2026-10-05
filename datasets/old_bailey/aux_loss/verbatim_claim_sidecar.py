#!/usr/bin/env python3
"""Build `final`, the chain's last step: `kind_balanced` with every `claim_real` statement set to the
verbatim record line.

kind_balanced (481 x {fabrication, claim_real, altered_real, paraphrase_real} per size) uses a
gpt-5.4-mini rewrite as its block-A TRUE class. This step changes only each claim_real statement
(and its slot in messages[-1]) back to its `source_evidence` line, adopting the base_origin
grandparent row's messages after proving that splicing at the `STATEMENT TO ASSESS:\\n"` anchor
reproduces them byte-for-byte. Row order, item_ids and pair_ids are kept, and unchanged rows
re-serialize to the parent's exact lines (P5b). No LLM, GPU or seed.

slot_safe is bypassed: 12 (4B) / 10 (8B) restored lines contain a raw `"`, as base_origin rendered
them. So substitute_statement cannot re-splice this stage, and make_train_sidecar_meta.py and
filter_aux_ce_sidecar.py cannot regenerate its _meta.json; this script writes it (chat_template_sha
copied from the parent, max_prompt_tokens recomputed: 4B 6810 -> 6851).

check_statement_balance.py fails on this variant by design (long verbatim TRUE vs short
fabrication); record it with --expect-fail --json-out, do not relax the gate.

Train sidecar only. The launcher's identity gate (ids == auxce_kind_balanced_<S>/sft_train.parquet;
frozen auxce_balanced_<S>/sft_val.parquet disjoint by item and game) is re-checked read-only. That
parquet holds the rewritten text, so sft_content_parity is recorded as false.

  .venv/bin/python datasets/old_bailey/aux_loss/verbatim_claim_sidecar.py --dry-run   # proofs only
  ... same without --dry-run, plus --sample-tsv <dir>/sample_rows.tsv --selfcheck
  .venv/bin/python datasets/old_bailey/aux_loss/verbatim_claim_sidecar.py --selfcheck-only
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    A_KINDS,
    ANCHOR,
    B_KINDS,
    DEFAULT_SRC,
    DEFAULT_TEMPLATE,
    GENERATED,
    KIND_BLOCK,
    SCHEMA_VERSION,
    TOKENIZER_PATH,
    check_render,
    emit_sample_tsv,
    git_commit,
    load_records,
    load_sidecar,
    norm,
    sha256_file,
    words,
)

GRANDPARENT_KINDS = ("control_real", "control_real_aug")
CHANGED_KEYS = {"messages", "statement", "rewrite_model", "verbatim_source"}


def die(msg: str):
    raise SystemExit(f"ERROR: {msg}")


# ------------------------------------------------------------------ transform


def splice_verbatim(messages, old: str, new: str):
    """rebalance_aux_ce_sidecar.substitute_statement without the quote guard (see module docstring).

    Splices behind ANCHOR, not str.replace, since the statement may also appear in the record block.
    The newline guard is kept.
    """
    msgs = [dict(m) for m in messages]
    content = msgs[-1]["content"]
    if msgs[-1]["role"] != "user":
        raise ValueError(f"probe chat must end on a user turn, got {msgs[-1]['role']!r}")
    if content.count(ANCHOR) != 1:
        raise ValueError(f"expected exactly 1 statement anchor, found {content.count(ANCHOR)}")
    j = content.index(ANCHOR) + len(ANCHOR)
    if content[j:j + len(old)] != old:
        raise ValueError("statement behind the anchor does not match the row's `statement` field")
    if content[j + len(old)] != '"':
        raise ValueError("statement behind the anchor is not closed by a quote")
    if "\n" in new or "\r" in new:
        raise ValueError(f"restored statement contains a newline: {new!r}")
    msgs[-1]["content"] = content[:j] + new + content[j + len(old):]
    return msgs


def serialize_row(row: dict) -> str:
    """The parent's serialization (--dump-train-sidecar), so unchanged rows are byte-identical."""
    return json.dumps(row, sort_keys=True)


def write_sidecar(path: Path, rows) -> None:
    """Write rows with GzipFile(mtime=0) so a rebuild is byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((serialize_row(r) + "\n").encode())


def raw_lines(path: Path) -> list:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def revert_row(row: dict, gp: dict, grandparent_tag: str) -> dict:
    """One claim_real row -> the same row carrying the grandparent's verbatim statement + messages."""
    iid = row["item_id"]
    if row["kind"] != "claim_real" or row["label"] != "true":
        die(f"{iid}: revert_row called on kind={row['kind']!r} label={row['label']!r}")
    if gp["kind"] not in GRANDPARENT_KINDS:
        die(f"{iid}: grandparent {gp['item_id']} has kind {gp['kind']!r}, expected one of {GRANDPARENT_KINDS}")
    if gp["label"] != "true":
        die(f"{iid}: grandparent {gp['item_id']} is not a TRUE row")
    if gp["game_id"] != row["game_id"] or gp["context_id"] != row["context_id"]:
        die(f"{iid}: grandparent {gp['item_id']} belongs to another game/context")
    if gp["statement"] != row["source_evidence"]:
        die(f"{iid}: source_evidence != {grandparent_tag} statement -- this row was not written from that record line")
    # Check the untouched turns separately; the splice below only rewrites the last turn.
    if [dict(m) for m in row["messages"][:-1]] != [dict(m) for m in gp["messages"][:-1]]:
        die(f"{iid}: context before the last turn differs from {gp['item_id']}")
    spliced = splice_verbatim(row["messages"], row["statement"], gp["statement"])
    if spliced != gp["messages"]:
        die(f"{iid}: anchor-splice does not reproduce {gp['item_id']} messages BYTE-FOR-BYTE")
    out = dict(row)
    out["messages"] = [dict(m) for m in gp["messages"]]   # grandparent's bytes, not the splice
    out["statement"] = gp["statement"]
    out["rewrite_model"] = None
    out["verbatim_source"] = f"{grandparent_tag}:{gp['item_id']}"
    return out


# ------------------------------------------------------------------ proofs


def verify_size(size: str, args, parent_meta: dict, gp_meta: dict, records: dict) -> dict:
    """Transform one size fully in memory; every proof is fatal. -> {rows, stats, meta_size}."""
    root = Path(args.src) / "variants"
    p_path = root / args.parent / size / "sidecar.jsonl.gz"
    g_path = root / args.grandparent / size / "sidecar.jsonl.gz"
    pm = parent_meta["sizes"].get(size) or die(f"parent _meta.json has no size {size!r}")
    gm = gp_meta["sizes"].get(size) or die(f"grandparent _meta.json has no size {size!r}")

    # P1 parent + grandparent integrity
    p_sha, g_sha = sha256_file(p_path), sha256_file(g_path)
    if p_sha != pm["sidecar_sha256"]:
        die(f"[{size}] {args.parent} sidecar sha {p_sha[:16]} != its _meta {pm['sidecar_sha256'][:16]} -- stale or edited parent")
    if g_sha != gm["sidecar_sha256"]:
        die(f"[{size}] {args.grandparent} sidecar sha {g_sha[:16]} != its _meta {gm['sidecar_sha256'][:16]} -- stale or edited grandparent")
    # P2 both were composed under the same chat template
    if pm["chat_template_sha"] != gm["chat_template_sha"]:
        die(f"[{size}] grandparent contexts were composed under a different chat template; a byte-exact restore would be meaningless")
    # P12 renders with the pinned TOKENIZER_PATH; the meta must name the same path.
    if pm["tokenizer_path"] != TOKENIZER_PATH[size]:
        die(f"[{size}] parent meta tokenizer_path {pm['tokenizer_path']} != the pinned TOKENIZER_PATH {TOKENIZER_PATH[size]}")

    parent_rows = load_sidecar(p_path)
    parent_again = load_sidecar(p_path)            # fresh objects so P5 is a real comparison
    gp_by_id = {}
    for r in load_sidecar(g_path):
        if r["item_id"] in gp_by_id:
            die(f"[{size}] duplicate item_id in {args.grandparent}: {r['item_id']!r}")
        gp_by_id[r["item_id"]] = r

    # P3 + transform
    new_rows, n_changed, claim_kinds = [], 0, Counter()
    for r in parent_rows:
        if r["kind"] == "claim_real":
            gp = gp_by_id.get(r.get("parent_item_id"))
            if gp is None:
                die(f"[{size}] {r['item_id']}: parent_item_id {r.get('parent_item_id')!r} not in {args.grandparent}")
            new_rows.append(revert_row(r, gp, args.grandparent))
            n_changed += 1
            claim_kinds[gp["kind"]] += 1
        else:
            new_rows.append(r)
    # P4 exactly the claim_real rows changed
    if n_changed != args.expect_per_kind:
        die(f"[{size}] {n_changed} claim_real rows changed, expected {args.expect_per_kind}")
    # P5 every non-claim row dict-equal to the parent; every claim row equal on every untouched key
    for r_new, r_par in zip(new_rows, parent_again):
        if r_new["item_id"] != r_par["item_id"]:
            die(f"[{size}] row order changed at {r_new['item_id']!r} / {r_par['item_id']!r}")
        if r_par["kind"] != "claim_real":
            if r_new != r_par:
                die(f"[{size}] {r_new['item_id']}: non-claim row changed (must be byte-identical to the parent)")
        else:
            if set(r_new) != set(r_par) | {"verbatim_source"}:
                die(f"[{size}] {r_new['item_id']}: unexpected key set {sorted(set(r_new) ^ set(r_par))}")
            for k in r_par:
                if k not in CHANGED_KEYS and r_new[k] != r_par[k]:
                    die(f"[{size}] {r_new['item_id']}: field {k!r} changed on a claim_real row")
            if r_new["statement"] == r_par["statement"]:
                die(f"[{size}] {r_new['item_id']}: claim_real statement unchanged (rewrite == record line?)")
    # P5b every unchanged row reproduces the parent's raw line byte-for-byte
    p_lines = raw_lines(p_path)
    if len(p_lines) != len(new_rows):
        die(f"[{size}] parent has {len(p_lines)} lines but {len(new_rows)} rows were loaded")
    n_raw_identical = 0
    for r_new, line in zip(new_rows, p_lines):
        if r_new["kind"] != "claim_real":
            if serialize_row(r_new) != line:
                die(f"[{size}] {r_new['item_id']}: re-serialized row is not the parent's raw line (serialization drift)")
            n_raw_identical += 1
        elif serialize_row(r_new) == line:
            die(f"[{size}] {r_new['item_id']}: claim_real line unchanged")
    # P6 id set + ORDER identical (the aux sampler indexes pools in file order)
    if [r["item_id"] for r in new_rows] != [r["item_id"] for r in parent_rows]:
        die(f"[{size}] row order or item_id set changed -- the one-factor claim would not hold")
    # P7 shape
    n = len(new_rows)
    labels = Counter(r["label"] for r in new_rows)
    kinds = Counter(r["kind"] for r in new_rows)
    expect_kinds = {k: args.expect_per_kind for k in KIND_BLOCK}
    if n != args.expect_items or n != pm["n_items"]:
        die(f"[{size}] {n} items, expected {args.expect_items} (parent meta {pm['n_items']})")
    if labels["true"] != labels["false"] or set(labels) != {"true", "false"} or labels["true"] != pm["n_true"]:
        die(f"[{size}] label split {dict(labels)} != 1:1 / parent meta {pm['n_true']}T/{pm['n_false']}F")
    if dict(kinds) != expect_kinds or dict(sorted(kinds.items())) != dict(sorted(pm["by_kind"].items())):
        die(f"[{size}] kind counts {dict(kinds)}, expected {expect_kinds} (parent meta {pm['by_kind']})")
    # P8 pair structure (mirrors make_train_sidecar_meta.verify_size step 3)
    by_pair: dict = {}
    for r in new_rows:
        if not r.get("pair_id"):
            die(f"[{size}] {r['item_id']!r} has no pair_id")
        by_pair.setdefault(r["pair_id"], []).append(r)
    block_cnt: Counter = Counter()
    for k, v in by_pair.items():
        if len(v) != 2:
            die(f"[{size}] pair {k!r} has {len(v)} rows, not 2")
        ks = {r["kind"] for r in v}
        if ks not in (A_KINDS, B_KINDS):
            die(f"[{size}] pair {k!r} has kinds {sorted(ks)} -- not a block-A or block-B pair")
        if {r["label"] for r in v} != {"true", "false"}:
            die(f"[{size}] pair {k!r} is not one TRUE + one FALSE")
        for r in v:
            block_cnt[(KIND_BLOCK[r["kind"]], r["label"])] += 1
    for blk in ("a", "b"):
        t, f = block_cnt[(blk, "true")], block_cnt[(blk, "false")]
        if not (t and f and t == f):
            die(f"[{size}] block {blk!r} label-imbalanced: {t}T/{f}F")
    # P9 every claim_real statement == source_evidence and is a Description: line of its own game
    n_exact, n_norm_only = 0, 0
    for r in new_rows:
        if r["kind"] != "claim_real":
            continue
        if r["statement"] != r["source_evidence"]:
            die(f"[{size}] {r['item_id']}: statement != source_evidence after the revert")
        lines = records.get(r["game_id"]) or records.get(int(r["game_id"])) or []
        if r["statement"] in {d.strip() for d in lines} or r["statement"] in lines:
            n_exact += 1
        elif norm(r["statement"]) in {norm(d) for d in lines}:
            n_norm_only += 1
        else:
            die(f"[{size}] {r['item_id']}: reverted statement is not a Description: line of its own game {r['game_id']}")
    # P10 ids unique, single channel, schema
    ids = [r["item_id"] for r in new_rows]
    if len(set(ids)) != len(ids):
        die(f"[{size}] duplicate item_id(s)")
    for r in new_rows:
        if r.get("term"):
            die(f"[{size}] {r['item_id']} carries `term` -- this is a single-channel sidecar")
        if r.get("schema_version") != SCHEMA_VERSION:
            die(f"[{size}] {r['item_id']} schema_version {r.get('schema_version')!r} != {SCHEMA_VERSION}")
    # P11 slot hygiene: record embedded quotes, gate newlines
    claims = [r for r in new_rows if r["kind"] == "claim_real"]
    n_quote = sum('"' in r["statement"] for r in claims)
    n_newline = sum(("\n" in r["statement"]) or ("\r" in r["statement"]) for r in claims)
    if n_newline:
        die(f"[{size}] {n_newline} restored statement(s) contain a newline")
    # P12 render under the pinned tokenizer; sha verified against the parent's, then COPIED
    cap = pm.get("max_prompt_tokens_cap", args.max_prompt_tokens)
    max_tok = check_render(new_rows, size, min(cap, args.max_prompt_tokens), pm["chat_template_sha"])
    # P13 the launcher's identity gates, read-only
    import pandas as pd
    sft_dir = Path(args.sft_dir_tpl.format(size=size))
    val_dir = Path(args.val_sft_dir_tpl.format(size=size))
    train_ids = set(pd.read_parquet(sft_dir / "sft_train.parquet", columns=["item_id"])["item_id"])
    if train_ids != set(ids):
        die(f"[{size}] sidecar ids != {sft_dir.name} sft_train ids (only-sidecar {len(set(ids) - train_ids)}, only-train {len(train_ids - set(ids))})")
    val = pd.read_parquet(val_dir / "sft_val.parquet", columns=["item_id", "game_id", "statement"])
    games = {int(r["game_id"]) for r in new_rows}
    n_val_item = len(set(val["item_id"]) & set(ids))
    n_val_game = len({int(g) for g in val["game_id"]} & games)
    n_val_stmt = len({norm(s) for s in val["statement"]} & {norm(r["statement"]) for r in new_rows})
    if n_val_item:
        die(f"[{size}] {n_val_item} frozen val item_id(s) found inside the sidecar")
    if n_val_game:
        die(f"[{size}] {n_val_game} frozen val GAME(s) found inside the sidecar (the holdout is by whole game)")

    stats = {
        "n_changed_rows": n_changed, "n_quote_in_slot": n_quote, "n_newline_in_slot": n_newline,
        "claim_parent_kinds": dict(sorted(claim_kinds.items())),
        "claim_statement_exact_record_line": n_exact, "claim_statement_norm_only_record_line": n_norm_only,
        "claim_words_mean": round(sum(words(r["statement"]) for r in claims) / len(claims), 2),
        "parent_claim_words_mean": round(sum(words(r["statement"]) for r in parent_rows
                                             if r["kind"] == "claim_real") / len(claims), 2),
        "n_unchanged_rows_raw_line_identical": n_raw_identical,
        "val_checks": {"item_overlap": n_val_item, "game_overlap": n_val_game, "statement_overlap": n_val_stmt},
    }
    if n_val_stmt:
        die(f"[{size}] {n_val_stmt} frozen-val statement(s) now appear verbatim in the sidecar")
    games_t = {int(r["game_id"]) for r in new_rows if r["label"] == "true"}
    games_f = {int(r["game_id"]) for r in new_rows if r["label"] == "false"}
    meta_size = {
        "n_items": n, "n_true": labels["true"], "n_false": labels["false"],
        "by_kind": dict(sorted(kinds.items())),
        "n_games": len(games), "n_games_with_true": len(games_t), "n_games_with_false": len(games_f),
        "tokenizer_path": pm["tokenizer_path"],
        "chat_template_sha": pm["chat_template_sha"],          # COPIED, verified by check_render
        "max_prompt_tokens": max_tok,                          # RECOMPUTED
        "max_prompt_tokens_cap": cap,
        "parent_max_prompt_tokens": pm["max_prompt_tokens"],
        "parent_sidecar_sha256": p_sha, "parent_n_items": len(parent_rows),
        "grandparent_sidecar_sha256": g_sha,
        "sidecar_sha256": None,                                # filled after the write
        "sft_dir": str(sft_dir), "n_val_frozen": int(len(val)),
        **stats,
    }
    print(f"[{size}] OK: {n} items ({labels['true']}T/{labels['false']}F), kinds {meta_size['by_kind']}, "
          f"{len(by_pair)} whole pairs, {len(games)} games; changed {n_changed} claim_real rows "
          f"(grandparent kinds {stats['claim_parent_kinds']}, {n_quote} with an embedded quote); "
          f"max_prompt_tokens {pm['max_prompt_tokens']} -> {max_tok}; claim words "
          f"{stats['parent_claim_words_mean']} -> {stats['claim_words_mean']}; ids == sft_train, val disjoint")
    return {"rows": new_rows, "meta_size": meta_size}


# ------------------------------------------------------------------ selfcheck


def selfcheck(out_root: Path, tag: str, sizes) -> int:
    """Re-validate through rl.aux_ce.load_state in both sampling modes and the submit gate's rules."""
    from transformers import AutoTokenizer

    from rl.aux_ce import DEFAULT_TERM, _prompt_ids, load_state

    meta = json.loads((out_root / tag / "_meta.json").read_text())
    ok = True
    for size in sizes:
        p = out_root / tag / size / "sidecar.jsonl.gz"
        sm = meta["sizes"][size]
        if not p.exists():
            print(f"[selfcheck] {size}: MISSING {p}")
            ok = False
            continue
        got = sha256_file(p)
        if got != sm["sidecar_sha256"]:
            print(f"[selfcheck] {size}: sha256 {got} != meta {sm['sidecar_sha256']}")
            ok = False
        rows = load_sidecar(p)
        lab = Counter(r["label"] for r in rows)
        kinds = Counter(r["kind"] for r in rows)
        if len(rows) != sm["n_items"] or lab["true"] != sm["n_true"] or lab["false"] != sm["n_false"] \
                or dict(sorted(kinds.items())) != dict(sorted(sm["by_kind"].items())):
            print(f"[selfcheck] {size}: counts {len(rows)}/{dict(lab)}/{dict(kinds)} != meta")
            ok = False
        if abs(lab["true"] - lab["false"]) > 0.01 * len(rows):
            print(f"[selfcheck] {size}: label split {dict(lab)} is >1% off 1:1")
            ok = False
        if any(r.get("term") for r in rows):
            print(f"[selfcheck] {size}: rows carry `term` -- this is a single-channel sidecar")
            ok = False
        ids = [r["item_id"] for r in rows]
        if len(set(ids)) != len(ids):
            print(f"[selfcheck] {size}: duplicate item_id(s)")
            ok = False
        for r in rows:
            roles = [m["role"] for m in r["messages"]]
            if roles[0] != "system" or roles[-1] != "user" or len(roles) % 2 != 0:
                print(f"[selfcheck] {size}: {r['item_id']} bad role pattern {roles}")
                ok = False
                break
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH[size])
        keys = ("AUX_CE", "AUX_CE_DATA", "AUX_CE_SAMPLING", "AUX_CE_KIND_BALANCE", "AUX_CE_MAX_PROMPT")
        for mode, kb in (("random", "1"), ("game", "0")):      # production config first
            saved = {k: os.environ.get(k) for k in keys}
            try:
                os.environ.update({"AUX_CE": "1", "AUX_CE_DATA": str(p), "AUX_CE_SAMPLING": mode,
                                   "AUX_CE_KIND_BALANCE": kb, "AUX_CE_MAX_PROMPT": "8192"})
                st = load_state(tok)
                if kb == "1":
                    pools = {k: len(v) for k, v in st.items_by_block_label.items()}
                    want = {(DEFAULT_TERM, blk, lb): sm["by_kind"]["claim_real"]
                            for blk in ("a", "b") for lb in ("true", "false")}
                    if pools != want:
                        print(f"[selfcheck] {size}/{mode}: block pools {pools} != {want}")
                        ok = False
                items = st.items_by_label["true"] + st.items_by_label["false"]
                longest = max(len(_prompt_ids(st, it)) for it in items)   # runtime cap assert, every row
                print(f"[selfcheck] {size}/{mode}/kindbal={kb}: rl.aux_ce.load_state OK ({len(rows)} rows, "
                      f"{dict(lab)}), every row tokenized, longest prompt {longest} tokens")
            except Exception as e:  # noqa: BLE001
                print(f"[selfcheck] {size}/{mode}: rl.aux_ce.load_state FAILED: {e}")
                ok = False
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
    print(f"[selfcheck] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", default="kind_balanced")
    ap.add_argument("--grandparent", default="base_origin")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="aux_ce_probe root (read)")
    ap.add_argument("--out", default=None, help="variants root (write); default <src>/variants")
    ap.add_argument("--sizes", nargs="+", default=["4B", "8B"])
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--sft-dir-tpl", default=str(GENERATED / "sft/auxce_kind_balanced_{size}"),
                    help="corpus whose sft_train ids the sidecar must equal (the launcher's identity gate)")
    ap.add_argument("--val-sft-dir-tpl", default=str(GENERATED / "sft/auxce_balanced_{size}"),
                    help="corpus whose frozen sft_val must stay disjoint by item AND game")
    ap.add_argument("--expect-items", type=int, default=1924)
    ap.add_argument("--expect-per-kind", type=int, default=481)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--sample-tsv", default=None)
    ap.add_argument("--dry-run", action="store_true", help="run every proof, write nothing")
    ap.add_argument("--selfcheck", action="store_true", help="after writing, re-validate through rl.aux_ce")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out) if args.out else src / "variants"
    if args.selfcheck_only:
        return selfcheck(out_root, args.tag, args.sizes)

    parent_meta_path = src / "variants" / args.parent / "_meta.json"
    gp_meta_path = src / "variants" / args.grandparent / "_meta.json"
    parent_meta = json.loads(parent_meta_path.read_text())
    gp_meta = json.loads(gp_meta_path.read_text())
    records, _bg = load_records(Path(args.template))

    built = {size: verify_size(size, args, parent_meta, gp_meta, records) for size in args.sizes}
    ref = built[args.sizes[0]]["meta_size"]
    for size in args.sizes[1:]:
        m = built[size]["meta_size"]
        if (m["n_items"], m["n_true"], m["n_false"], m["by_kind"]) != (ref["n_items"], ref["n_true"], ref["n_false"], ref["by_kind"]):
            die(f"sizes not equalized: {args.sizes[0]}={ref['n_items']}/{ref['by_kind']} vs {size}={m['n_items']}/{m['by_kind']}")
    print(f"[verbatim] per-kind counts identical across {args.sizes}: {ref['by_kind']}")
    if args.dry_run:
        print("[verbatim] --dry-run: every proof passed; nothing written")
        return 0

    for size in args.sizes:
        sp = out_root / args.tag / size / "sidecar.jsonl.gz"
        write_sidecar(sp, built[size]["rows"])
        built[size]["meta_size"]["sidecar_sha256"] = sha256_file(sp)
        print(f"[verbatim] {size}: wrote {len(built[size]['rows'])} rows -> {sp} sha256 {built[size]['meta_size']['sidecar_sha256']}")

    meta = {
        "schema_version": parent_meta.get("schema_version", SCHEMA_VERSION),
        "arm": parent_meta.get("arm"),
        "split_file": parent_meta.get("split_file"),
        "git_commit": git_commit(_REPO),
        "variant": {
            "tag": args.tag, "parent": args.parent, "grandparent": args.grandparent,
            "builder": "datasets/old_bailey/aux_loss/verbatim_claim_sidecar.py",
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv[1:],
            "serialization": ("json.dumps(row, sort_keys=True) (ensure_ascii=True) + newline, GzipFile(mtime=0): "
                              "the parent's exact regime, so every unchanged row is a byte-identical LINE of "
                              "the parent file (proven per row at build time)"),
            "design": ("kind_balanced with every claim_real statement (block-A TRUE) set to the byte-identical "
                       "evidence Description: line it was written from (the base_origin grandparent row's messages "
                       "and statement are adopted verbatim); fabrication / altered_real / paraphrase_real rows are "
                       "byte-identical LINES of kind_balanced's file, item_ids, pair_ids and row order are identical, "
                       "per-kind counts are identical across sizes. slot_safe is BYPASSED for the restored lines "
                       "that contain a raw double quote (they are exactly the base_origin bytes), so this stage and "
                       "anything derived from it can never be re-spliced by substitute_statement, and "
                       "check_statement_balance.py fails on it BY DESIGN (record with --expect-fail). TRAIN sidecar "
                       "only: no val split, no SFT parquet."),
            "parent_meta": str(parent_meta_path), "grandparent_meta": str(gp_meta_path),
            "sft_content_parity": False,
            "sft_content_parity_note": ("ids equal auxce_kind_balanced_<size>/sft_train.parquet (the launcher's "
                                        "identity gate) but that parquet holds the REWRITTEN claim_real text"),
        },
        "sizes": {size: built[size]["meta_size"] for size in args.sizes},
    }
    mp = out_root / args.tag / "_meta.json"
    mp.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[verbatim] -> {mp}")
    if args.sample_tsv:
        emit_sample_tsv(Path(args.sample_tsv), {s: built[s]["rows"] for s in args.sizes})
        print(f"[verbatim] -> {args.sample_tsv}")
    print(f"[verbatim] NEXT: datasets/old_bailey/aux_loss/check_statement_balance.py --variant {args.tag} --expect-fail --json-out ... "
          "(the gate FAILS by design; record the numbers)")
    if args.selfcheck:
        return selfcheck(out_root, args.tag, args.sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
