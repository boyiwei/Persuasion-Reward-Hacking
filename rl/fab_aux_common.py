"""Shared helpers for the aux-CE sidecar chain (`datasets/old_bailey/aux_loss/`).

The chain's builders run by path, so nothing under `datasets/` can be imported; every name more
than one of them needs lives here: the probe-prompt anchor, the item KINDS of a pair block, default
data roots, shared text helpers, and sidecar IO / provenance / render checks. No model or GPU is
used; `check_render` lazily loads a tokenizer.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Every artifact the chain reads or writes lives under the one generated root.
GENERATED = REPO / "datasets/old_bailey/_generated"
DEFAULT_SRC = GENERATED / "aux_ce_probe"
DEFAULT_VARIANTS = DEFAULT_SRC / "variants"
DEFAULT_TEMPLATE = (REPO / "datasets/old_bailey/processed/full"
                    / "old_bailey_revised_independent_full.json")

SCHEMA_VERSION = 1
SIZES = ("4B", "8B")

# Anchor before the statement under assessment. Substitute at this offset, not with str.replace:
# a control_real statement also appears verbatim in the message's `information:` list, and a blind
# replace would rewrite the evidence it is checked against.
ANCHOR = 'STATEMENT TO ASSESS:\n"'

# Tokenizers the sidecar contexts are token-checked with, per policy size, under MODELS_DIR.
MODELS_DIR = os.environ.get("MODELS_DIR") or str(REPO / "models")
TOKENIZER_PATH = {
    "4B": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507",
    "8B": f"{MODELS_DIR}/Qwen3-8B",
}

# Matched-pair blocks: A = real claim vs fabrication, B = paraphrase vs altered record line.
# AUX_CE_KIND_BALANCE draws one TRUE and one FALSE from the same block, so block membership is part
# of the data contract.
A_KINDS = frozenset(("fabrication", "claim_real"))
B_KINDS = frozenset(("altered_real", "paraphrase_real"))
KIND_BLOCK = {"fabrication": "a", "claim_real": "a", "altered_real": "b", "paraphrase_real": "b"}

DEF_VERBATIM_RUN = 8            # a shared run of >= this many words counts as "verbatim"

_WS = re.compile(r"\s+")


# ------------------------------------------------------------------ text helpers


def norm(s) -> str:
    return _WS.sub(" ", str(s or "").strip().lower())


def words(s: str) -> int:
    return len(str(s).split())


def description(ev: str):
    """The `Description:` line of one evidence item -- the claim text the chain compares."""
    m = re.search(r"^Description:\s*(.+)$", str(ev), flags=re.M)
    return m.group(1).strip() if m else None


def longest_shared_run(a: str, b: str, n: int = DEF_VERBATIM_RUN) -> bool:
    """True if a and b share any run of >= n consecutive words (the non-verbatim guard)."""
    aw, bw = norm(a).split(), norm(b).split()
    if len(aw) < n or len(bw) < n:
        return norm(a) in norm(b) or norm(b) in norm(a)
    grams = {" ".join(bw[i:i + n]) for i in range(len(bw) - n + 1)}
    return any(" ".join(aw[i:i + n]) in grams for i in range(len(aw) - n + 1))


# ------------------------------------------------------------------ provenance


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return ""


def builder_basename(builder: str) -> str:
    """File name of a `_meta.json` `builder` string (which also carries flags). Variants are
    recognised by this name, not the path, so a builder can move."""
    return Path(str(builder).split()[0]).name if str(builder).strip() else ""


# ------------------------------------------------------------------ sidecar IO


def load_sidecar(path: Path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_records(template: Path):
    """game_id -> ([evidence Description lines], background)."""
    recs, bgs = {}, {}
    for g in json.load(open(template)):
        priv = ((g.get("params") or {}).get("private") or {})
        recs[g["id"]] = [d for d in (description(e) for e in (priv.get("information") or [])) if d]
        pub = ((g.get("params") or {}).get("public") or {})
        bgs[g["id"]] = str(pub.get("game_background") or "").strip()
    return recs, bgs


def chat_template_sha(tok) -> str:
    return hashlib.sha256((tok.chat_template or "").encode()).hexdigest()


def check_render(rows, size, max_prompt_tokens, parent_tpl_sha):
    """Re-render every row; enforce the token cap and verify the tokenizer has not drifted.

    The sha is copied from the parent into _meta.json (recomputing would let a drifted tokenizer
    re-bless the file) and verified here.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH[size])
    got = chat_template_sha(tok)
    if got != parent_tpl_sha:
        raise SystemExit(
            f"ERROR [{size}]: tokenizer chat_template sha {got} != parent _meta {parent_tpl_sha}.\n"
            f"       tokenizer={TOKENIZER_PATH[size]}\n"
            "       The parent contexts were composed under a different template; rewriting on top "
            "of them would bake in a mis-rendered context.")
    max_tok, over = 0, []
    for r in rows:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, enable_thinking=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        n = len(ids)
        if n > max_prompt_tokens:
            over.append((r["item_id"], n))
        max_tok = max(max_tok, n)
    if over:
        raise SystemExit(f"ERROR [{size}]: {len(over)} row(s) exceed --max-prompt-tokens "
                         f"{max_prompt_tokens}, e.g. {over[:3]}")
    return max_tok


def emit_sample_tsv(path: Path, per_size_rows, n=50, seed=0):
    """A stratified eyeball sample -- the backstop the self-check gate cannot replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["size\tkind\tlabel\tgame_id\twords\tstatement\tsource_evidence"]
    for size, rows in per_size_rows.items():
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        per = max(1, n // (2 * max(1, len(by_kind))))
        for kind in sorted(by_kind):
            pool = sorted(by_kind[kind], key=lambda r: r["item_id"])
            random.Random(f"{seed}|tsv|{size}|{kind}").shuffle(pool)
            for r in pool[:per]:
                s = r["statement"].replace("\t", " ").replace("\n", " ")
                src = str(r.get("source_evidence", "")).replace("\t", " ").replace("\n", " ")
                lines.append(f"{size}\t{kind}\t{r['label']}\t{r['game_id']}\t"
                             f"{words(r['statement'])}\t{s}\t{src}")
    path.write_text("\n".join(lines) + "\n")
