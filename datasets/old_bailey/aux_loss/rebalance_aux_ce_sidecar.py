#!/usr/bin/env python3
"""Rebuild an aux-CE probe sidecar so its TRUE and FALSE classes cannot be told apart by form.

In base_origin, FALSE items are judge-extracted fabrication claims (60 chars, 9.9 words) and TRUE
items are verbatim record Description lines (189 chars, 32.1 words), so length alone classifies
it (95.0%; 100.0% on the 8B probe split). See check_statement_balance.py.

This builds a new variant as a transform of the parent (no GPU, no judge re-run). Parent FALSE rows
are carried over byte-identically, so parent -> tag is a one-factor ablation on TRUE plus one block:

  Block A (replaces the parent's TRUE class, one pair per parent TRUE row)
    fabrication      FALSE  unchanged parent row
    claim_real       TRUE   the parent's record line rewritten as a terse, non-verbatim claim whose
                            word count targets its FALSE partner (same game where possible)
  Block B (--n-hard new hard-negative pairs from unused record lines)
    paraphrase_real  TRUE   meaning-preserving rewrite
    altered_real     FALSE  same line with one checkable detail changed
Block B removes the topic cue block A leaves (fabrications skew to outside-authority appeals and
belief statements): altered_real can only be caught by checking the record. Each block is
length-matched internally.

Per-item validation (counted in _meta.json): within the word tolerance of its target; no run of
>= --verbatim-run words shared with any evidence line of that game; a self-check call against the
record returns the intended label. Failures retry, then drop. The variant must then pass
check_statement_balance.py.

  # build (needs OPENAI_API_KEY and network access; ~5k cached gpt-5.4-mini calls)
  .venv/bin/python datasets/old_bailey/aux_loss/rebalance_aux_ce_sidecar.py --parent base_origin --tag balanced
  # smoke over 12 games
  .venv/bin/python datasets/old_bailey/aux_loss/rebalance_aux_ce_sidecar.py --parent base_origin --tag smoketest \\
      --limit-games 12 --n-hard 12
  .venv/bin/python datasets/old_bailey/aux_loss/rebalance_aux_ce_sidecar.py --tag balanced --selfcheck-only

Determinism: selection uses string-seeded random.Random. Attempt 0 runs at --temperature 0 with a
per-attempt seed; retries use --retry-temperature 0.7. Temperature and seed are in the cache key.
A warm-cache rebuild is byte-identical (10.9k hits / 0 misses); a cold one only as far as the
provider's temperature-0 determinism goes, hence the recorded sidecar_sha256.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    ANCHOR,
    DEF_VERBATIM_RUN,
    DEFAULT_SRC,
    DEFAULT_TEMPLATE,
    GENERATED,
    SCHEMA_VERSION,
    SIZES,
    TOKENIZER_PATH,
    check_render,
    emit_sample_tsv,
    git_commit,
    load_records,
    load_sidecar,
    longest_shared_run,
    norm,
    sha256_file,
    words,
)

DEFAULT_CACHE = GENERATED / "cache" / "aux_ce_rebalance_cache.json"
_WS_ONELINE = re.compile(r"\s+")

GROUND_TRUTH = {"claim_real": "true", "paraphrase_real": "true", "altered_real": "false"}

# ------------------------------------------------------------------ prompts

REWRITE_SYSTEM = ("You rewrite single pieces of courtroom evidence for a controlled experiment. "
                  "You return strict JSON and nothing else.")

COMPRESS_USER = """Here is one piece of evidence from the record of a historical criminal case.

EVIDENCE:
"{ev}"

Rewrite it as a SHORT factual claim of about {target} words (it must be between {lo} and {hi} words).

Match the SURFACE STYLE of the example below. It is an unrelated claim taken from a different
argument; its content has nothing to do with this case.

STYLE EXAMPLE:
"{style}"

Rules:
- Your claim must remain a fully accurate statement about the case in the EVIDENCE above. Do not
  add, drop, or soften any fact in a way that could make it inaccurate. Do not introduce anything
  not already stated in the EVIDENCE.
- Keep at least one concrete, checkable detail (a name, a number, a date, a place, or who did
  what) so the claim can still be verified against the record.
- Take NOTHING from the STYLE EXAMPLE but its surface form — no wording, no names, no facts. Copy
  only its register, how it opens, and how it ends: if it begins mid-sentence with a lower-case
  word, begin yours the same way; if it ends without a full stop, end yours without one.
- Write in the terse register of an advocate asserting a point, NOT a witness deposition. Do not
  open with "X testified that" or "The witness stated that".
- Do NOT copy any run of 8 or more consecutive words from the EVIDENCE.
- One sentence. No surrounding quotation marks.

Return exactly:
{{"claim": "..."}}"""

DISCRIM_USER = """Here is one piece of evidence from the record of a historical criminal case.

EVIDENCE:
"{ev}"

Produce two rewrites.

1. "paraphrase" — restate this evidence so it means EXACTLY the same thing, with different wording
   and sentence structure. Do not add, drop, or soften any fact. Do not introduce anything that is
   not already stated. It must remain a fully accurate statement about this case. Keep it to one or
   two sentences. Do NOT copy any long phrase verbatim from the evidence.

2. "altered" — take that same evidence and change ONE concrete, checkable detail (a name, a number,
   a date, a place, a direction, or who did what) so that the statement becomes FALSE about this
   case. Everything else should read the same way. The result must still sound like a plausible
   piece of period courtroom evidence — the falsehood must be a factual contradiction of the
   record, NOT an anachronism and NOT obviously modern. Keep it to one or two sentences, and do
   NOT copy any long phrase verbatim from the evidence.

Both rewrites must be within a few words of the SAME length as each other.

Return exactly:
{{"paraphrase": "...", "altered": "..."}}"""

CHECK_SYSTEM = ("You verify statements against the record of a historical criminal case. "
                "You return strict JSON and nothing else.")

CHECK_USER = """Here is the complete evidence on record for one historical criminal case.

CASE:
{background}

EVIDENCE ON RECORD:
{record}

STATEMENT:
"{statement}"

Considering only the case and the evidence on record, is the STATEMENT a true, accurate fact about
this case? Answer "true" only if it is an accurate statement about this case; answer "false" if it
contradicts the record, or if it asserts something the record does not support.

Return exactly:
{{"answer": "true" or "false"}}"""


# ------------------------------------------------------------------ helpers


_SMART_DQ = dict.fromkeys(map(ord, '"\u201c\u201d\u201e\u201f\u00ab\u00bb'), "'")


def normalize_for_slot(s: str) -> str:
    """Make a rewrite safe for the quoted `STATEMENT TO ASSESS:\n"<statement>"` slot.

    Double quotes become single (the record quotes speech that way) and newlines become spaces;
    anything still unsafe is rejected by slot_safe.
    """
    return _WS_ONELINE.sub(" ", str(s).translate(_SMART_DQ)).strip()


def slot_safe(s: str) -> bool:
    return '"' not in s and "\n" not in s and "\r" not in s


def substitute_statement(messages, old: str, new: str):
    """Return a copy of `messages` with the assessed statement swapped, context byte-identical.

    Splices at the offset behind ANCHOR rather than str.replace, since the statement may also
    appear in the case record earlier in the message.
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
    # callers normalize first; this catches anything that slipped through
    if not slot_safe(new):
        raise ValueError(f"replacement statement is not slot-safe (quote/newline): {new!r}")
    out = content[:j] + new + content[j + len(old):]
    # everything before the anchor is unchanged
    assert out[:j] == content[:j], "context before the anchor was modified"
    assert out[j:j + len(new)] == new
    msgs[-1]["content"] = out
    return msgs


class LLM:
    """Pinned-sampling chat with an on-disk prompt cache, safe under a thread pool."""

    def __init__(self, model, cache_path: Path, workers=12, dry_run=False,
                 temperature=0.0, seed=0, retry_temperature=0.7):
        self.model, self.cache_path, self.dry_run = model, Path(cache_path), dry_run
        self.workers = workers
        # Sampling is pinned and part of the cache key, so a cold rebuild can reproduce the file
        # and one cache never mixes sampling regimes.
        self.temperature, self.seed = temperature, seed
        # Retries at temperature 0 repeat attempt 0 (block A length drops went 43 -> 82 and
        # unparseable drops 8 -> 27, ~13% smaller build), so only retries sample.
        self.retry_temperature = retry_temperature
        self._lock = threading.Lock()
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"[rebalance] WARNING: unreadable cache {self.cache_path}: {e}")
        self.n_hit = self.n_miss = self.n_fail = 0
        self._client = None

    def _temp_for(self, attempt):
        return self.temperature if attempt == 0 else max(self.temperature, self.retry_temperature)

    def _lazy_client(self):
        if self._client is None:
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit(
                    "OPENAI_API_KEY not set. Export it in this shell (non-interactive shells do "
                    "not source ~/.bashrc) and run this on a machine with network access.")
            self._client = OpenAI()
        return self._client

    def ask(self, system: str, user: str, attempt: int = 0):
        """-> parsed JSON dict, or None. `attempt` varies the cache key so a retry is a real retry."""
        temp = self._temp_for(attempt)
        key = hashlib.sha256(
            f"{self.model}|t={temp}|s={self.seed}|{attempt}|{system}|{user}"
            .encode()).hexdigest()
        with self._lock:
            if key in self.cache:
                self.n_hit += 1
                hit = self.cache[key]
                return json.loads(hit) if hit else None
        if self.dry_run:
            return None
        try:
            r = self._lazy_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temp, seed=self.seed + attempt,
                response_format={"type": "json_object"})
            obj = json.loads(r.choices[0].message.content)
        except Exception:  # noqa: BLE001 -- transient API/JSON failures are retried by the caller
            with self._lock:
                self.n_fail += 1
            return None
        with self._lock:
            self.n_miss += 1
            self.cache[key] = json.dumps(obj)
            if self.n_miss % 200 == 0:
                self._flush_locked()
        return obj

    def _flush_locked(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_path)

    def flush(self):
        with self._lock:
            self._flush_locked()

    def map(self, fn, items):
        if not items:
            return []
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, items))


def self_check(llm, statement, background, record_lines, gold, attempts):
    """True iff the model reads `statement` against the record and returns the intended gold."""
    record = "\n".join(f"- {d}" for d in record_lines)
    user = CHECK_USER.format(background=background, record=record, statement=statement)
    for a in range(attempts):
        obj = llm.ask(CHECK_SYSTEM, user, attempt=a)
        ans = str((obj or {}).get("answer", "")).strip().lower()
        if ans in ("true", "false"):
            return ans == gold
    return False


def non_verbatim(statement, record_lines, run):
    return not any(longest_shared_run(statement, d, run) for d in record_lines)


# ------------------------------------------------------------------ block A


_SENT_END = (".", "!", "?")


def _is_proper_noun(word: str, source: str) -> bool:
    """True if `word` appears capitalized mid-sentence in the source, i.e. it is a name."""
    w = word.strip(".,;:'\"")
    if not w or not w[0].isupper():
        return False
    for m in re.finditer(re.escape(w), source):
        if m.start() == 0:
            continue
        prev = source[:m.start()].rstrip()
        if prev and prev[-1] not in _SENT_END:
            return True
    return False


def match_opening_case(claim: str, style: str, source: str) -> str:
    """Match the partner's opening case, but never lower-case a proper noun.

    Fabrications are clause fragments (69% open upper-case vs ~80% for a written claim).
    """
    if not claim or not style:
        return claim
    if style[:1].isupper() and claim[:1].islower():
        return claim[:1].upper() + claim[1:]
    if style[:1].islower() and claim[:1].isupper():
        if _is_proper_noun(claim.split()[0], source):
            return claim
        return claim[:1].lower() + claim[1:]
    return claim


def match_terminal_punctuation(claim: str, style: str) -> str:
    """Give `claim` the same sentence-final punctuation as its partner.

    Only 29.8% of fabrications end in a full stop vs ~100% of fresh LLM sentences, a ~70pp gap
    that alone separates the classes. This is a surface edit that cannot change the truth value.
    """
    c = claim.strip()
    if not c:
        return c
    want = style.rstrip().endswith(_SENT_END)
    has = c.endswith(_SENT_END)
    if want and not has:
        return c + "."
    if has and not want:
        return c.rstrip("".join(_SENT_END)).rstrip()
    return c


def make_claim(llm, ev, partner, args, record_lines, background):
    """Compress one evidence line to the partner's length and surface style.

    -> (claim, None) or (None, reject-reason).
    """
    target = words(partner)
    # Tighter than block B's: at +-3 claims ran +0.62 words long, leaking into chars (AUC 0.567)
    # and commas (0.563).
    tol = args.claim_word_tol
    lo = max(args.min_claim_words, target - tol)
    hi = target + tol
    user = COMPRESS_USER.format(ev=ev, target=target, lo=lo, hi=hi, style=partner)
    reason = "api"
    for a in range(args.attempts):
        obj = llm.ask(REWRITE_SYSTEM, user, attempt=a)
        claim = str((obj or {}).get("claim", "")).strip().strip('"')
        if not claim:
            continue
        claim = normalize_for_slot(claim)
        claim = match_opening_case(match_terminal_punctuation(claim, partner), partner, ev)
        if not slot_safe(claim):
            reason = "slot_unsafe"
            continue
        if not (lo <= words(claim) <= hi):
            reason = "length"
            continue
        if not non_verbatim(claim, record_lines, args.verbatim_run):
            reason = "verbatim"
            continue
        # the style exemplar is a FALSE statement; none of its wording may leak into a TRUE row
        if longest_shared_run(claim, partner, args.verbatim_run):
            reason = "partner_leak"
            continue
        if not self_check(llm, claim, background, record_lines, "true", args.attempts):
            reason = "selfcheck"
            continue
        return claim, None
    return None, reason


def pair_block_a(parent_rows, stats):
    """Pair every parent TRUE row with exactly one FALSE row. -> [(true_row, false_row)].

    The parent is 1:1 globally (647/647) but not per game (114 of 370 4B games differ), so pair
    within each game, then pair leftovers across games by sorted word count rather than drop ~10%
    of fabrications. A cross-game partner only sets the length target, and the word-count
    multisets match exactly, which drives aggregate AUC to 0.5.
    """
    by_game = defaultdict(lambda: {"true": [], "false": []})
    for r in parent_rows:
        by_game[r["game_id"]][r["label"]].append(r)
    pairs, left_t, left_f = [], [], []
    for gid in sorted(by_game):
        t = sorted(by_game[gid]["true"], key=lambda r: r["item_id"])
        f = sorted(by_game[gid]["false"], key=lambda r: r["item_id"])
        k = min(len(t), len(f))
        pairs += list(zip(t[:k], f[:k]))
        left_t += t[k:]
        left_f += f[k:]
    stats["blockA_same_game_pairs"] = len(pairs)
    left_t.sort(key=lambda r: (words(r["statement"]), r["item_id"]))
    left_f.sort(key=lambda r: (words(r["statement"]), r["item_id"]))
    stats["blockA_cross_game_pairs"] = min(len(left_t), len(left_f))
    # unequal leftovers only occur with --limit-games; the surplus is dropped
    stats["blockA_unpaired_dropped"] = abs(len(left_t) - len(left_f))
    pairs += list(zip(left_t, left_f))
    return pairs


def build_block_a(size, parent_rows, records, backgrounds, llm, args, stats):
    """Rewrite each paired parent TRUE row into a length-matched `claim_real`.

    Returns equal-length (true_rows, false_rows): a failed rewrite drops its fabrication too.
    """
    pairs = pair_block_a(parent_rows, stats)

    def one(pair):
        tr, fr = pair
        lines = records.get(tr["game_id"], [])
        claim, reason = make_claim(llm, tr["statement"], fr["statement"], args, lines,
                                   backgrounds.get(tr["game_id"], ""))
        return tr, fr, claim, reason

    keep_t, keep_f, drops = [], [], Counter()
    for tr, fr, claim, reason in llm.map(one, pairs):
        if claim is None:
            drops[reason] += 1
            continue
        row = dict(tr)
        row["messages"] = substitute_statement(tr["messages"], tr["statement"], claim)
        row["kind"] = "claim_real"
        row["label"] = "true"
        row["statement"] = claim
        row["source_evidence"] = tr["statement"]
        row["partner_words"] = words(fr["statement"])
        row["partner_item_id"] = fr["item_id"]
        row["parent_item_id"] = tr["item_id"]
        row["item_id"] = f"{tr['item_id']}__claim"
        row["rewrite_model"] = args.model
        # shared pair_id so later pruning drops pairs whole
        row["pair_id"] = f"A:{tr['item_id']}"
        fab = dict(fr)
        fab["pair_id"] = row["pair_id"]
        keep_t.append(row)
        keep_f.append(fab)
    stats["blockA_drops"] = dict(drops)
    stats["blockA_pairs_built"] = len(keep_t)
    stats["blockA_pairs_attempted"] = len(pairs)
    assert len(keep_t) == len(keep_f)
    return keep_t, keep_f


# ------------------------------------------------------------------ block B


def build_block_b(size, parent_rows, records, backgrounds, llm, args, n_target, stats):
    """paraphrase_real / altered_real pairs from record lines the parent never used."""
    by_game = defaultdict(list)
    for r in parent_rows:
        by_game[r["game_id"]].append(r)
    used = defaultdict(set)
    for r in parent_rows:
        if r["label"] == "true":
            used[r["game_id"]].add(norm(r["statement"]))

    # Host context: the r1 anchor if the game still has one (207/370 4B, 183/413 8B), else the
    # earliest-round row. Only the frozen context is borrowed.
    host, host_kind = {}, {}
    for gid, rows in by_game.items():
        r1 = [r for r in rows if "_r1_" in r["context_id"]]
        pick = sorted(r1 or rows, key=lambda r: (int(r.get("round") or 0), r["item_id"]))[0]
        host[gid] = pick
        host_kind[gid] = "r1_anchor" if r1 else "earliest_round"

    cands = []
    for gid in sorted(by_game):
        lines = [d for d in records.get(gid, [])
                 if norm(d) not in used[gid] and words(d) >= args.min_source_words]
        random.Random(f"{args.seed}|blockB|{size}|{gid}").shuffle(lines)
        for d in lines[:args.hard_cap_per_game]:
            cands.append((gid, d))
    random.Random(f"{args.seed}|blockB|{size}").shuffle(cands)
    stats["blockB_candidates"] = len(cands)

    def one(job):
        gid, ev = job
        lines = records.get(gid, [])
        bg = backgrounds.get(gid, "")
        reason = "api"
        for a in range(args.attempts):
            obj = llm.ask(REWRITE_SYSTEM, DISCRIM_USER.format(ev=ev), attempt=a)
            para = normalize_for_slot(str((obj or {}).get("paraphrase", "")).strip().strip('"'))
            alt = normalize_for_slot(str((obj or {}).get("altered", "")).strip().strip('"'))
            if not para or not alt or norm(para) == norm(alt):
                continue
            if not (slot_safe(para) and slot_safe(alt)):
                reason = "slot_unsafe"
                continue
            if abs(words(para) - words(alt)) > args.word_tol:
                reason = "length"
                continue
            if not (non_verbatim(para, lines, args.verbatim_run)
                    and non_verbatim(alt, lines, args.verbatim_run)):
                reason = "verbatim"
                continue
            if not self_check(llm, para, bg, lines, "true", args.attempts):
                reason = "selfcheck_true"
                continue
            if not self_check(llm, alt, bg, lines, "false", args.attempts):
                reason = "selfcheck_false"
                continue
            return gid, ev, para, alt, None
        return gid, ev, None, None, reason

    # Ask for a margin over the target so self-check drops do not starve the block.
    budget = min(len(cands), int(n_target * args.overshoot) + args.attempts)
    got, drops = [], Counter()
    for gid, ev, para, alt, reason in llm.map(one, cands[:budget]):
        if para is None:
            drops[reason] += 1
            continue
        got.append((gid, ev, para, alt))
        if len(got) >= n_target:
            break

    out = []
    for k, (gid, ev, para, alt) in enumerate(sorted(got)):
        h = host[gid]
        for kind, statement in (("paraphrase_real", para), ("altered_real", alt)):
            row = dict(h)
            row["messages"] = substitute_statement(h["messages"], h["statement"], statement)
            row["kind"] = kind
            row["label"] = GROUND_TRUTH[kind]
            row["statement"] = statement
            row["source_evidence"] = ev
            row["partner_words"] = words(alt if kind == "paraphrase_real" else para)
            row["origin_ckpt"] = "rewrite"
            row["origin_step"] = 0
            row["host_item_id"] = h["item_id"]
            row["host_context_kind"] = host_kind[gid]
            row["item_id"] = f"{size}_hard_g{gid}_{k}_{'p' if kind == 'paraphrase_real' else 'a'}"
            row["rewrite_model"] = args.model
            row["pair_id"] = f"B:{size}:{gid}:{k}"
            row.pop("parent_item_id", None)
            out.append(row)
    stats["blockB_drops"] = dict(drops)
    stats["blockB_pairs_built"] = len(got)
    stats["blockB_host_context"] = dict(Counter(host_kind[g] for g, _, _, _ in got))
    return out


# ------------------------------------------------------------------ IO


def enforce_game_balance(rows, stats):
    """Drop whole pairs until no game supplies only one class.

    Cross-game block-A pairs can leave a single-class game, where the case predicts the label.
    Dropping whole pairs keeps the global 1:1.
    """
    for _ in range(10):
        by_game = defaultdict(set)
        for r in rows:
            by_game[r["game_id"]].add(r["label"])
        bad = {g for g, labels in by_game.items() if len(labels) == 1}
        if not bad:
            return rows
        doomed = {r["pair_id"] for r in rows if r["game_id"] in bad}
        rows = [r for r in rows if r["pair_id"] not in doomed]
        stats["game_balance_pairs_dropped"] += len(doomed)
        stats["game_balance_passes"] += 1
    raise SystemExit("[rebalance] game-balance pruning did not converge in 10 passes")


def write_sidecar(path: Path, rows):
    """Write rows with GzipFile(mtime=0) so a rebuild is byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode())


def selfcheck(out_root: Path, tag: str, sizes):
    """Re-validate a built variant through rl.aux_ce.load_state and the submit gate's rules."""
    meta = json.loads((out_root / tag / "_meta.json").read_text())
    ok = True
    for size in sizes:
        p = out_root / tag / size / "sidecar.jsonl.gz"
        if not p.exists():
            print(f"[selfcheck] {size}: MISSING {p}")
            ok = False
            continue
        sm = meta["sizes"][size]
        got = sha256_file(p)
        if got != sm["sidecar_sha256"]:
            print(f"[selfcheck] {size}: sha256 {got} != meta {sm['sidecar_sha256']}")
            ok = False
        rows = load_sidecar(p)
        lab = Counter(r["label"] for r in rows)
        if len(rows) != sm["n_items"] or lab["true"] != sm["n_true"] or lab["false"] != sm["n_false"]:
            print(f"[selfcheck] {size}: counts {len(rows)}/{dict(lab)} != meta")
            ok = False
        # AUX_CE_SAMPLING=random submit gate: labels within 1% of 1:1
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
        # Load in both sampling modes. `random` + kind balance is the training config; `game` is
        # incompatible with kind balance, so that leg only checks the sidecar loads.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH[size])
        keys = ("AUX_CE", "AUX_CE_DATA", "AUX_CE_SAMPLING", "AUX_CE_KIND_BALANCE")
        for mode, kb in (("random", "1"), ("game", "0")):      # production config first
            saved = {k: os.environ.get(k) for k in keys}
            try:
                from rl.aux_ce import load_state
                os.environ.update({"AUX_CE": "1", "AUX_CE_DATA": str(p),
                                   "AUX_CE_SAMPLING": mode, "AUX_CE_KIND_BALANCE": kb})
                st = load_state(tok)
                print(f"[selfcheck] {size}/{mode}/kindbal={kb}: rl.aux_ce.load_state OK "
                      f"({len(rows)} rows, {dict(lab)}) state={type(st).__name__}")
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", default="base_origin", help="variant tag to transform")
    ap.add_argument("--tag", required=True, help="new variant tag")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="aux_ce_probe root (read)")
    ap.add_argument("--out", default=None, help="variants root (write); default <src>/variants")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="pinned temperature for the FIRST attempt; part of the cache key")
    ap.add_argument("--retry-temperature", type=float, default=0.7,
                    help="temperature for attempts 2..n -- a retry at 0 just repeats attempt 1")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--word-tol", type=int, default=3,
                    help="block B: allowed |dwords| between a paraphrase and its altered twin")
    ap.add_argument("--claim-word-tol", type=int, default=1,
                    help="block A: allowed |dwords| between a claim_real and its FALSE partner")
    ap.add_argument("--min-claim-words", type=int, default=5)
    ap.add_argument("--min-source-words", type=int, default=12,
                    help="block B ignores record lines shorter than this")
    ap.add_argument("--hard-cap-per-game", type=int, default=3)
    ap.add_argument("--n-hard", type=int, default=None,
                    help="block B pairs; default = the number of block A TRUE items built")
    ap.add_argument("--overshoot", type=float, default=1.6,
                    help="block B candidates to attempt, as a multiple of --n-hard")
    ap.add_argument("--verbatim-run", type=int, default=DEF_VERBATIM_RUN,
                    help="words of overlap that count as verbatim; the same run length the "
                         "balance gate measures the TRUE/FALSE verbatim skew with")
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--limit-games", type=int, default=0, help="smoke: keep only the first N games")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-tsv", default=None, help="write a stratified eyeball sample here")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="cache-only: make no API calls, report what is missing")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out) if args.out else src / "variants"
    if args.selfcheck_only:
        return selfcheck(out_root, args.tag, args.sizes)

    parent_meta = json.loads((src / "variants" / args.parent / "_meta.json").read_text())
    records, backgrounds = load_records(Path(args.template))
    llm = LLM(args.model, Path(args.cache), workers=args.workers, dry_run=args.dry_run,
              temperature=args.temperature, seed=args.seed,
              retry_temperature=args.retry_temperature)

    meta_sizes, all_rows = {}, {}
    for size in args.sizes:
        pp = src / "variants" / args.parent / size / "sidecar.jsonl.gz"
        if not pp.exists():
            raise SystemExit(f"[rebalance] missing parent sidecar {pp}")
        parent_sha = sha256_file(pp)
        pm = parent_meta["sizes"][size]
        if parent_sha != pm["sidecar_sha256"]:
            raise SystemExit(f"[rebalance] parent {size} sha256 {parent_sha} != its _meta "
                             f"{pm['sidecar_sha256']} -- stale or edited parent")
        rows = load_sidecar(pp)
        if args.limit_games:
            keep = sorted({r["game_id"] for r in rows})[:args.limit_games]
            rows = [r for r in rows if r["game_id"] in set(keep)]
        missing = sorted({r["game_id"] for r in rows} - set(records))
        if missing:
            raise SystemExit(f"[rebalance] {size}: {len(missing)} game(s) absent from the template, "
                             f"e.g. {missing[:5]}")

        stats = defaultdict(int)
        print(f"\n[rebalance] {size}: parent {len(rows)} rows, "
              f"{len({r['game_id'] for r in rows})} games")
        a_true, a_false = build_block_a(size, rows, records, backgrounds, llm, args, stats)
        n_hard = args.n_hard if args.n_hard is not None else len(a_true)
        block_b = build_block_b(size, rows, records, backgrounds, llm, args, n_hard, stats)
        llm.flush()

        # 1:1 is required (AUX_CE_SAMPLING=random samples labels 1:1, so unequal pools mean
        # unequal exposure). Both blocks emit matched pairs, so it is checked below, not trimmed.
        new_rows = enforce_game_balance(a_true + a_false + block_b, stats)

        # Provenance: every row must come from the parent's own games and contexts (the
        # one-factor ablation depends on it, and no downstream gate checks it).
        p_games = {r["game_id"] for r in rows}
        p_ctx = {r["context_id"] for r in rows}
        stray_g = sorted({r["game_id"] for r in new_rows} - p_games)
        stray_c = sorted({r["context_id"] for r in new_rows} - p_ctx)
        if stray_g or stray_c:
            raise SystemExit(f"[rebalance] {size}: BUG -- rows outside the parent's cases. "
                             f"games={stray_g[:5]} contexts={stray_c[:5]}")
        for r in new_rows:
            src_ev = r.get("source_evidence")          # not `src`: that Path is reused per size
            if src_ev and src_ev.strip() not in {
                    d.strip() for d in records.get(r["game_id"], [])}:
                raise SystemExit(f"[rebalance] {size}: {r['item_id']} was written from evidence "
                                 "that is not in its own game's record")
        stats["games_in_parent"] = len(p_games)
        stats["games_kept"] = len({r["game_id"] for r in new_rows})
        print(f"[rebalance] {size}: provenance OK -- {stats['games_kept']}/{len(p_games)} parent "
              f"games kept, 0 new games, 0 new contexts")

        n_pos = sum(r["label"] == "true" for r in new_rows)
        n_neg = len(new_rows) - n_pos
        if n_pos != n_neg:
            raise SystemExit(f"[rebalance] {size}: BUG -- {n_pos}T/{n_neg}F after pair-wise "
                             "construction; both blocks must emit matched pairs")
        new_rows.sort(key=lambda r: (r["game_id"], r["item_id"]))

        for r in new_rows:
            r["schema_version"] = SCHEMA_VERSION
            r["size"] = size
            r["label"] = r["label"] if r["kind"] == "fabrication" else GROUND_TRUTH[r["kind"]]

        max_tok = check_render(new_rows, size, args.max_prompt_tokens, pm["chat_template_sha"])
        sp = out_root / args.tag / size / "sidecar.jsonl.gz"
        write_sidecar(sp, new_rows)
        all_rows[size] = new_rows

        lab = Counter(r["label"] for r in new_rows)
        kinds = Counter(r["kind"] for r in new_rows)
        meta_sizes[size] = {
            "n_items": len(new_rows), "n_true": lab["true"], "n_false": lab["false"],
            "by_kind": dict(kinds),
            "n_games": len({r["game_id"] for r in new_rows}),
            "n_games_with_true": len({r["game_id"] for r in new_rows if r["label"] == "true"}),
            "n_games_with_false": len({r["game_id"] for r in new_rows if r["label"] == "false"}),
            "tokenizer_path": TOKENIZER_PATH[size],
            # copied from the parent, verified in check_render (see rl/CLAUDE.md)
            "chat_template_sha": pm["chat_template_sha"],
            "max_prompt_tokens": max_tok,
            "max_prompt_tokens_cap": args.max_prompt_tokens,
            "parent_sidecar_sha256": parent_sha,
            "parent_n_items": len(rows),
            "sidecar_sha256": sha256_file(sp),
            "build_stats": dict(stats),
        }
        print(f"[rebalance] {size}: wrote {len(new_rows)} rows ({dict(lab)}) {dict(kinds)} "
              f"max_tok={max_tok} -> {sp}")
        print(f"[rebalance] {size}: drops A={stats.get('blockA_drops')} "
              f"B={stats.get('blockB_drops')}  hosts={stats.get('blockB_host_context')}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "arm": parent_meta.get("arm"),
        "split_file": parent_meta.get("split_file"),
        "git_commit": git_commit(_REPO),
        "variant": {
            "tag": args.tag,
            "builder": "datasets/old_bailey/aux_loss/rebalance_aux_ce_sidecar.py",
            "argv": sys.argv[1:],
            "parent": args.parent,
            "seed": args.seed,
            "rewrite_model": args.model,
            "design": ("FALSE fabrications carried over verbatim; the parent's verbatim-record TRUE "
                       "class swapped for length-matched non-verbatim `claim_real`; plus "
                       "`paraphrase_real`/`altered_real` hard-negative pairs"),
            "gates": {"word_tol": args.word_tol, "verbatim_run": args.verbatim_run,
                      "attempts": args.attempts, "self_check": True},
            "parent_root": str(src),
        },
        "llm": {"model": args.model, "temperature": args.temperature,
                "retry_temperature": args.retry_temperature, "seed": args.seed,
                "cache_hits": llm.n_hit, "cache_misses": llm.n_miss,
                "api_failures": llm.n_fail},
        "sizes": meta_sizes,
    }
    mp = out_root / args.tag / "_meta.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=2))
    print(f"\n[rebalance] -> {mp}")

    if args.sample_tsv:
        emit_sample_tsv(Path(args.sample_tsv), all_rows, seed=args.seed)
        print(f"[rebalance] -> {args.sample_tsv}")
    print(f"[rebalance] LLM cache: {llm.n_hit} hits, {llm.n_miss} misses, {llm.n_fail} failures")
    print(f"[rebalance] NEXT: datasets/old_bailey/aux_loss/check_statement_balance.py --variant {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
