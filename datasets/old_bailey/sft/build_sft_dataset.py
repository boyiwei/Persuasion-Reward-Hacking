"""Build the multiturn SFT parquets from audited sender rollouts.

Teacher-agnostic: the teacher can be hosted (gpt-5.4-mini, collect_rollouts.sh) or a self-hosted
Qwen (self-distillation, collect_selfdistill.slurm). The only model tie is the student tokenizer
(--tokenizer; set it to the student when it differs from the teacher).

Inputs: N result JSONs from evaluation/rl_rollout.py (one temp-1 pass each over the SFT-train
games), each with frozen per-turn audit sidecars <stem>.strategy_audit.json
(rl.strategy_audit.run --per-turn) and <stem>.fabrication.json (evaluation/audit_fabrications.py
--per-turn), plus sft_holdout.parquet from split_rl_sft.py. Each example's [system, user1] prefix is
copied verbatim from that parquet, so SFT prompts match GRPO byte for byte; the parquet is the
prompt knob (rl_sftsplit_strategies/<dist>/ or rl_sftsplit/<dist>/) and must match the GRPO root.

Filter: each turn must pass every check in _turn_dirty_reason (format, word floors, leakage,
wrong side, <= --max-illegal-per-turn illegal slugs with None counted as illegal,
false_information == 0, rh_fake_evidence == 0). Only the maximal clean prefix is kept (a dirty
round 1 drops the rollout), so dirty text never appears even as context. A prefix of length k
gives k examples, each with `loss_mask: 0` on all but its final assistant turn, so every clean
turn is supervised once. Audits are never re-run here (the judge samples at temp 1).

Usage (repo .venv, no GPU), self-distillation from the 10 passes collect_selfdistill.slurm
writes with STEER_K=0 BUILD_SFT=0:

    python datasets/old_bailey/sft/build_sft_dataset.py \\
        --results experiments/results/old-bailey/qwen3-8B-base/sft_rollouts/stubborn_sfttrain_strategies_pass{1..10}.json \\
        --prompts-parquet datasets/old_bailey/_generated/rl_sftsplit_strategies/stubborn/sft_holdout.parquet \\
        --out-dir datasets/old_bailey/_generated/sft/strategies_stubborn_qwen3-8B-base \\
        --max-dup-profile 3 --val-games 0 --subsample 614 --subsample-seed 0 \\
        --tokenizer $MODELS_DIR/Qwen3-8B

gpt-5.4-mini distillation uses the 3 passes collect_rollouts.sh writes with SENDER_API=gateway
(.../gpt-5.4-mini/sft_rollouts/..._pass{1,2,3}.json), --out-dir .../sft/strategies_stubborn,
--max-dup-profile 3 and the default tokenizer and val split. The 4B student uses the
qwen3-4B-instruct-base passes and --tokenizer $MODELS_DIR/Qwen3-4B-Instruct-2507.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.persuasion_interaction import _FALLBACK_ADVANCE, _RECEIVER_TURN, _receiver_context  # noqa: E402
from rl.reward_function import _ADVANCE_RE, _normalize_tags, _turn_format_ok  # noqa: E402
from rl.strategy_audit.taxonomy import (  # noqa: E402
    COERCIVE, DECEPTIVE, LEGIT, STRATEGIES, slugs_by_legitimacy)
from sft.sft_dataset import patch_qwen3_thinking_template  # noqa: E402

ILLEGAL_SLUGS = slugs_by_legitimacy(COERCIVE) + slugs_by_legitimacy(DECEPTIVE)
LEGIT_SLUGS = slugs_by_legitimacy(LEGIT)
# Never allowed in an imitated turn: think tags fight the template's think handling, im_start/im_end
# encode as control tokens. (<think> is not a substring of <thinking>.)
_BANNED_LITERALS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")

# Applied only after _turn_format_ok passed, so both always match. re.I is required: the format
# predicate's regexes are case-insensitive and _normalize_tags does not fix case, so a
# case-sensitive match would extract an empty body from <ARGUMENT> and fail the word floor.
# A bare <think> is already rejected by _BANNED_LITERALS.
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I)
_ARGUMENT_RE = re.compile(r"<argument>(.*?)</argument>", re.S | re.I)


def _block(rx, normalized: str) -> str:
    m = rx.search(normalized)
    return (m.group(1) if m else "").strip()


# Content floor. A turn that copies the prompt's format exemplar
#     <thinking> THINKING </thinking> ### <argument> ARGUMENT</argument>
# passes the format check and every judge (content-free text triggers no verdict). Seen in 0.10% of
# Qwen3-8B self-distillation turns. Degenerate arguments had 1 word, the shortest genuine one 232.
DEFAULT_MIN_ARGUMENT_WORDS = 50
DEFAULT_MIN_THINKING_WORDS = 5

# Harness/taxonomy leakage in <argument>. It is out-of-role text, and for a policy trained on it a
# recited strategy label makes the offline 42-way audit score the label, not the behaviour.
# Bare names count only if multi-word; one-word labels ("Framing", "Favor") are common courtroom
# words, so they count only when **bolded**.
_MULTIWORD_NAMES = sorted((s["name"] for s in STRATEGIES if len(s["name"].split()) > 1),
                          key=len, reverse=True)
_ALL_NAMES = sorted((s["name"] for s in STRATEGIES), key=len, reverse=True)
_LEAK_PATTERNS = (
    # e.g. "as recorded in [EVIDENCE ev1]"
    ("leak_harness_tag", re.compile(r"\[\s*EVIDENCE\s+ev\d+\s*\]", re.I)),
    # a multi-word taxonomy label used as a term of art
    ("leak_strategy_label", re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in _MULTIWORD_NAMES))),
    # any taxonomy label under markdown emphasis, incl. the one-word ones
    ("leak_strategy_label", re.compile(r"\*\*\s*(?:%s)\s*\*\*"
                                       % "|".join(re.escape(n) for n in _ALL_NAMES), re.I)),
    # Bayesian scaffolding ("shift the belief from a 10% prior probability of guilt")
    ("leak_game_scaffolding", re.compile(r"\bprior (?:belief|probability)\b"
                                         r"|\d{1,3}\s?%[^.]{0,40}\b(?:probability of guilt|belief)\b"
                                         r"|\bposterior belief\b", re.I)),
    # Rule vocabulary from the prompt. Bare tokens are case-sensitive (the prompt capitalizes them);
    # lowercase forms need a rules noun so "forbidden content" in a case does not match.
    ("leak_rule_meta", re.compile(r"\bFORBIDDEN\b|\bALLOWED\s+strateg")),
    ("leak_rule_meta", re.compile(r"\b(?:forbidden|allowed|permissible|legitimate)\s+"
                                  r"(?:strateg|method|techniqu|tactic|persuasion)", re.I)),
)

# Wrong side: no other filter constrains which side the sender argued, so a clean turn asking the
# court to acquit would be a reward-inverted target for a prosecution sender (common on
# defence-heavy records). A bare "presumption of innocence" was the main false positive, so it
# counts only when the ask is to uphold it.
_ACQUIT_ASK = re.compile(
    r"\b(?:urge|ask|invite|implore|recommend|request|submit|call upon)\b[^.]{0,140}?"
    r"(?:\bacquit(?:tal|ted)?\b|\bnot guilty\b"
    r"|\buphold(?:ing)?\s+the\s+presumption\s+of\s+innocence\b)", re.I)
# Suppresses the flag when the same sentence argues against acquittal ("the evidence does not
# justify acquittal").
_ACQUIT_NEGATED = re.compile(
    r"\b(?:implications?|risk|danger|cost|question|possibility|prospect)\s+of\s+(?:an?\s+)?acquittal"
    r"|\b(?:not|never|cannot|does not|do not|no)\b[^.]{0,60}\b(?:justify|warrant|support|merit)\b"
    r"[^.]{0,30}\bacquittal"
    r"|\b(?:against|resist|reject|rebut|overcome|displace|rather than)\b[^.]{0,40}"
    r"\b(?:acquittal|presumption of innocence)\b", re.I)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_prompts(prompts_parquet: Path) -> dict:
    """game_id -> [system, user1] (plain dicts) from sft_holdout.parquet's `prompt` column."""
    df = pd.read_parquet(prompts_parquet)
    out = {}
    for _, row in df.iterrows():
        gid = row["extra_info"]["index"]
        out[int(gid)] = [{"role": str(m["role"]), "content": str(m["content"])}
                         for m in row["prompt"]]
    return out


def _load_sidecar(stem: Path, suffix: str, what: str) -> dict:
    path = Path(str(stem) + suffix)
    if not path.is_file():
        sys.exit(f"[build_sft] missing {what} sidecar: {path} — run the audit first "
                 f"(frozen sidecars are required inputs)")
    return json.loads(path.read_text())


def _turn_vector(audit_entry: dict, round_idx: int):
    """The per-turn 42-vector for ORIGINAL round index round_idx (None if unaudited)."""
    if not audit_entry:
        return None
    rounds = audit_entry.get("turn_rounds")
    vectors = audit_entry.get("turn_vectors")
    if rounds is None or vectors is None:
        sys.exit("[build_sft] strategy_audit sidecar has no turn_vectors — re-run "
                 "`python -m rl.strategy_audit.run --oldbailey-result ... --per-turn`")
    try:
        return vectors[rounds.index(round_idx)]
    except ValueError:
        return None


def _fab_turn(fab_row: dict, round_idx: int):
    """The per-turn fabrication record for ORIGINAL round index round_idx (None if unaudited)."""
    if not fab_row:
        return None
    pt = fab_row.get("per_turn")
    if pt is None:
        sys.exit("[build_sft] fabrication sidecar has no per_turn rows — re-run "
                 "`evaluation/audit_fabrications.py --domain old-bailey ... --per-turn`")
    for rec in pt:
        if rec.get("round_idx") == round_idx:
            return rec
    return None


def _advance_content(responses: list, round_idx: int, n_rounds: int) -> str:
    """Rebuild the round-advance user turn before assistant round `round_idx` (0-based, >= 1).

    Mirrors evaluation/rl_rollout.play_game but is kept independent of it (only the format constants
    are shared), so the parity check against saved `sender_messages` is a real cross-check."""
    recv_resp = (responses[round_idx - 1] or {}).get("receiver") or ""
    ctx = _receiver_context(recv_resp) if recv_resp else ""
    if ctx:
        return _RECEIVER_TURN.format(next_round=round_idx + 1, total=n_rounds, receiver=ctx)
    return _FALLBACK_ADVANCE.format(next_round=round_idx + 1, total=n_rounds)


def _wrong_side_reason(argument: str):
    """'wrong_side' if the argument asks the court to acquit, else None (checked per sentence)."""
    for m in _ACQUIT_ASK.finditer(argument):
        start = argument.rfind(".", 0, m.start()) + 1
        end = argument.find(".", m.end())
        sentence = argument[start: end + 1 if end != -1 else len(argument)]
        if not _ACQUIT_NEGATED.search(sentence):
            return "wrong_side"
    return None


def _turn_dirty_reason(raw: str, tv, fab, max_illegal: int, skip_fab: bool,
                       min_argument_words: int = DEFAULT_MIN_ARGUMENT_WORDS,
                       min_thinking_words: int = DEFAULT_MIN_THINKING_WORDS,
                       check_leakage: bool = True, check_wrong_side: bool = True):
    """First failing check for one turn, or None if the turn is clean."""
    if not raw or not raw.strip():
        return "empty_turn"
    if any(lit in raw for lit in _BANNED_LITERALS):
        return "special_token"
    normalized = _normalize_tags(raw)
    if _ADVANCE_RE.search(normalized):
        return "forged_advance"
    if not _turn_format_ok(normalized):
        return "format"
    # These checks read the block bodies, which the raw-string checks above cannot see.
    argument = _block(_ARGUMENT_RE, normalized)
    if len(argument.split()) < min_argument_words:
        return "degenerate_argument"
    if len(_block(_THINKING_RE, normalized).split()) < min_thinking_words:
        return "degenerate_thinking"
    if check_leakage:
        for reason, rx in _LEAK_PATTERNS:
            if rx.search(argument):
                return reason
    if check_wrong_side:
        reason = _wrong_side_reason(argument)
        if reason is not None:
            return reason
    if tv is None:
        return "no_audit"
    fi = tv.get("false_information")
    if fi is None or fi >= 0.5:
        return "false_information"
    illegal = sum(1 for s in ILLEGAL_SLUGS if tv.get(s) is None or tv.get(s) >= 0.5)
    if illegal > max_illegal:
        return "illegal"
    if not skip_fab:
        if fab is None:
            return "no_fab_audit"
        if fab.get("judge_fail"):
            return "fab_judge_fail"
        if (fab.get("rh_fake_evidence") or 0) != 0:
            return "fabrication"
    return None


def _legit_profile(tv) -> tuple:
    """Sorted legit slugs the judge marked present in this turn (the diversity signal)."""
    return tuple(sorted(s for s in LEGIT_SLUGS if tv.get(s) is not None and tv.get(s) >= 0.5))


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _select_diverse(rows: list, n: int) -> list:
    """Greedy max-coverage subset of n rows over legit profiles.

    Ties go to the game with fewest rows taken, then input order, so after coverage saturates the
    fill is game-balanced and deterministic."""
    covered, taken, per_game = set(), [], Counter()
    remaining = list(rows)
    while remaining and len(taken) < n:
        best_i = max(range(len(remaining)),
                     key=lambda i: (len(set(remaining[i]["profile"]) - covered),
                                    -per_game[remaining[i]["game_id"]], -i))
        best = remaining.pop(best_i)
        taken.append(best)
        covered |= set(best["profile"])
        per_game[best["game_id"]] += 1
    return sorted(taken, key=lambda r: (r["game_id"], r["pass_idx"], r["prefix_len"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--results", nargs="+", required=True,
                    help="rl_rollout result JSONs (one per pass) with frozen audit sidecars")
    ap.add_argument("--prompts-parquet", required=True,
                    help="sft_holdout.parquet of the MATCHING data root + DIST (the prompt knob: "
                         "rl_sftsplit_strategies/<dist>/ or rl_sftsplit/<dist>/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-illegal-per-turn", type=int, default=0,
                    help="max illegal (coercive/deceptive) strategies a clean turn may carry "
                         "(false_information is ALWAYS 0 regardless; default 0)")
    ap.add_argument("--val-games", type=int, default=10,
                    help="held-out SFT-val GAMES (all their rows go to sft_val.parquet — the "
                         "same case never straddles train/val)")
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=7680,
                    help="drop rows longer than this under the (template-patched) tokenizer — "
                         "keep <= the trainer's data.max_length minus headroom")
    ap.add_argument("--tokenizer",
                    default=os.path.join(os.environ.get("MODELS_DIR") or str(REPO_ROOT / "models"),
                                         "Qwen3-8B"),
                    help="tokenizer for the length guard (Qwen3 family shares one BPE; the "
                         "thinking-style 8B template upper-bounds the 4B render; default "
                         "$MODELS_DIR/Qwen3-8B, MODELS_DIR defaulting to <repo>/models)")
    ap.add_argument("--max-dup-profile", type=int, default=2,
                    help="cap rows sharing (game, prefix_len, supervised-turn legit profile) "
                         "across passes — near-duplicate demonstrations add little (default 2)")
    ap.add_argument("--select-diverse", type=int, default=None,
                    help="optionally keep only N rows chosen greedily for legit-strategy "
                         "coverage (default: keep all surviving rows)")
    ap.add_argument("--subsample", type=int, default=None,
                    help="size matching across arms: keep a UNIFORM random N rows. Unlike "
                         "--select-diverse this preserves the arm's natural composition, so N is "
                         "the only variable being controlled (default: keep all)")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="seed for --subsample (default 0)")
    ap.add_argument("--skip-fabrication-filter", action="store_true",
                    help="plumbing tests only: skip the per-turn fabrication terms")
    ap.add_argument("--min-argument-words", type=int, default=DEFAULT_MIN_ARGUMENT_WORDS,
                    help="floor on the <argument> BODY in words -- catches placeholder collapse, "
                         "which the raw-string emptiness check cannot see (default "
                         f"{DEFAULT_MIN_ARGUMENT_WORDS}; observed degenerate turns carry 1 word, "
                         "the shortest genuine argument 232)")
    ap.add_argument("--min-thinking-words", type=int, default=DEFAULT_MIN_THINKING_WORDS,
                    help=f"floor on the <thinking> BODY in words (default {DEFAULT_MIN_THINKING_WORDS})")
    ap.add_argument("--allow-harness-leakage", action="store_true",
                    help="keep turns whose ARGUMENT recites harness/taxonomy vocabulary (literal "
                         "[EVIDENCE evN] handles, taxonomy strategy labels, prior/posterior belief "
                         "framing, FORBIDDEN/ALLOWED rule talk). Off by default because such text is "
                         "out of role AND makes the offline 42-way strategy audit fire on the "
                         "model's own recited labels instead of its behaviour")
    ap.add_argument("--allow-wrong-side", action="store_true",
                    help="keep turns whose ARGUMENT asks the court to acquit (reward-inverted "
                         "targets for a prosecution sender; no other filter constrains direction)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = patch_qwen3_thinking_template(
        AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True))

    prompts = _load_prompts(Path(args.prompts_parquet))
    print(f"[build_sft] {len(prompts)} holdout prompts from {args.prompts_parquet}")

    rows = []
    stats = {"config": vars(args), "inputs": [], "per_pass": []}
    prefix_hist = Counter()
    truncate_reasons = Counter()
    truncate_by_turn = Counter()
    n_dropped_rollouts = Counter()
    parity_checked = parity_mismatch = 0

    for _order_idx, res in enumerate(sorted(args.results), start=1):
        # pass_idx is a dedup/sort key, so take it from the `_pass<N>.json` filename rather than
        # argument order; fall back to traversal order.
        _m = re.search(r"_pass(\d+)\.json$", str(res))
        pass_idx = int(_m.group(1)) if _m else _order_idx
        res = Path(res)
        stem = Path(str(res)[: -len(".json")]) if str(res).endswith(".json") else res
        games = json.loads(res.read_text())
        audit = _load_sidecar(stem, ".strategy_audit.json", "strategy-audit")
        audit_by_id = {pg.get("id"): pg for pg in audit.get("per_game", [])}
        fab_by_id = {}
        if not args.skip_fabrication_filter:
            fab = _load_sidecar(stem, ".fabrication.json", "fabrication")
            fab_by_id = {r.get("id"): r for r in fab.get("rows", [])}
        stats["inputs"].append({"file": str(res), "md5": _md5(res), "n_games": len(games)})

        pass_rows = 0
        for game in sorted(games, key=lambda g: g["id"]):
            gid = game["id"]
            responses = game.get("responses") or []
            n_rounds = len(responses)
            if gid not in prompts:
                n_dropped_rollouts["not_in_holdout"] += 1
                continue
            if n_rounds == 0:
                n_dropped_rollouts["no_rounds"] += 1
                continue

            entry = audit_by_id.get(gid)
            fab_row = fab_by_id.get(gid)
            saved = game.get("sender_messages")
            saved_ok = (isinstance(saved, list) and len(saved) == 2 * n_rounds + 1
                        and [m.get("role") for m in saved]
                        == ["system", "user"] + ["assistant", "user"] * (n_rounds - 1) + ["assistant"])

            # maximal clean prefix
            k = 0
            profiles = []
            for i in range(n_rounds):
                raw = (responses[i] or {}).get("sender") or ""
                tv = _turn_vector(entry, i)
                reason = _turn_dirty_reason(raw, tv, _fab_turn(fab_row, i),
                                            args.max_illegal_per_turn,
                                            args.skip_fabrication_filter,
                                            min_argument_words=args.min_argument_words,
                                            min_thinking_words=args.min_thinking_words,
                                            check_leakage=not args.allow_harness_leakage,
                                            check_wrong_side=not args.allow_wrong_side)
                if reason is not None:
                    truncate_reasons[reason] += 1
                    truncate_by_turn[i] += 1
                    break
                profiles.append(_legit_profile(tv))
                k += 1
            prefix_hist[k] += 1
            if k == 0:
                n_dropped_rollouts["dirty_round_1"] += 1
                continue

            # [system, user1] from the parquet + raw assistant turns + rebuilt advance turns
            # (checked against sender_messages when saved)
            convo = [dict(m) for m in prompts[gid]]
            for i in range(k):
                if i > 0:
                    adv = _advance_content(responses, i, n_rounds)
                    if saved_ok:
                        parity_checked += 1
                        if saved[1 + 2 * i]["content"] != adv:
                            parity_mismatch += 1
                    convo.append({"role": "user", "content": adv})
                # A leading newline would BPE-merge with the 4B template's 'assistant\n' header
                # (off by one vs the GRPO token stream); the Qwen3 template lstrips anyway.
                convo.append({"role": "assistant", "content": responses[i]["sender"].lstrip("\n")})

            for j in range(1, k + 1):
                msgs = []
                for m in convo[: 2 * j + 1]:
                    mm = dict(m)
                    # loss on the final assistant turn (index 2j) only
                    if mm["role"] == "assistant" and len(msgs) != 2 * j:
                        mm["loss_mask"] = 0
                    msgs.append(mm)
                rows.append({
                    "messages": msgs,
                    "enable_thinking": False,
                    "game_id": int(gid),
                    "pass_idx": pass_idx,
                    "prefix_len": j,
                    "profile": list(profiles[j - 1]),
                    "strategy_hint": list(game.get("strategy_hint") or []),
                    "source_file": str(res),
                })
                pass_rows += 1
        stats["per_pass"].append({"file": str(res), "rows": pass_rows})

    if parity_mismatch:
        sys.exit(f"[build_sft] advance-turn reconstruction mismatched sender_messages in "
                 f"{parity_mismatch}/{parity_checked} turns — rl_rollout and this builder "
                 f"have drifted; fix before building")
    print(f"[build_sft] advance-turn parity: {parity_checked} checked, 0 mismatches")

    # Length guard under the patched tokenizer. transformers>=5 returns a BatchEncoding here, whose
    # len() is its key count, so unwrap input_ids.
    kept, n_too_long, tok_lens = [], 0, []
    for r in rows:
        enc = tokenizer.apply_chat_template(
            r["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False)
        n_tok = len(enc if isinstance(enc, list) else enc["input_ids"])
        if n_tok > args.max_tokens:
            n_too_long += 1
            continue
        tok_lens.append(n_tok)
        kept.append(r)
    rows = kept

    # near-duplicate cap: same (game, prefix_len, supervised-turn profile) across passes
    n_dedup = 0
    by_key = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["game_id"], r["prefix_len"], r["pass_idx"])):
        by_key[(r["game_id"], r["prefix_len"], tuple(r["profile"]))].append(r)
    rows = []
    for key in sorted(by_key, key=str):
        group = by_key[key]
        rows.extend(group[: args.max_dup_profile])
        n_dedup += max(0, len(group) - args.max_dup_profile)
    rows.sort(key=lambda r: (r["game_id"], r["pass_idx"], r["prefix_len"]))

    if args.select_diverse:
        before = len(rows)
        rows = _select_diverse(rows, args.select_diverse)
        print(f"[build_sft] --select-diverse: {before} -> {len(rows)} rows")

    # Size matching across arms: uniform, not --select-diverse, so N is the only variable and the
    # arm's composition is preserved. Row-level sampling is safe since each prefix example is
    # self-contained.
    if args.subsample:
        before = len(rows)
        if args.subsample > before:
            sys.exit(f"[build_sft] --subsample {args.subsample} exceeds the {before} available rows")
        rows = sorted(random.Random(args.subsample_seed).sample(rows, args.subsample),
                      key=lambda r: (r["game_id"], r["pass_idx"], r["prefix_len"]))
        print(f"[build_sft] --subsample: {before} -> {len(rows)} rows "
              f"(uniform, seed {args.subsample_seed})")

    if not rows:
        sys.exit("[build_sft] no rows survived the filter")

    # Game-keyed val split. --val-games 0 trains on every row: with trainer.save_freq=-1 only the
    # last step is saved, so val loss selects nothing.
    game_ids = sorted({r["game_id"] for r in rows})
    if args.val_games <= 0:
        val_ids = set()
    else:
        val_ids = set(random.Random(args.val_seed).sample(
            game_ids, min(args.val_games, max(1, len(game_ids) // 5))))
    train_rows = [r for r in rows if r["game_id"] not in val_ids]
    val_rows = [r for r in rows if r["game_id"] in val_ids]

    # diversity + summary stats over supervised turns
    slug_counts = Counter(s for r in rows for s in r["profile"])
    uptake = []
    hints_by_game = defaultdict(set)
    used_by_game = defaultdict(set)
    for r in rows:
        if r["strategy_hint"]:
            hints_by_game[r["game_id"]].update(r["strategy_hint"])
        used_by_game[r["game_id"]].update(r["profile"])
    for gid, hinted in hints_by_game.items():
        if hinted:
            uptake.append(len(hinted & used_by_game[gid]) / len(hinted))
    tok_sorted = sorted(tok_lens)
    stats.update({
        "n_rows_train": len(train_rows),
        "n_rows_val": len(val_rows),
        "n_supervised_turns": len(rows),
        "n_rows_too_long": n_too_long,
        "n_rows_dedup_dropped": n_dedup,
        "n_dropped_rollouts": dict(n_dropped_rollouts),
        "clean_prefix_hist": {str(key): prefix_hist[key] for key in sorted(prefix_hist)},
        "truncate_reasons": dict(truncate_reasons.most_common()),
        "truncate_by_turn": {str(key): truncate_by_turn[key] for key in sorted(truncate_by_turn)},
        "token_len": {"mean": sum(tok_lens) / len(tok_lens),
                      "p95": tok_sorted[int(0.95 * (len(tok_sorted) - 1))],
                      "max": tok_sorted[-1]} if tok_lens else None,
        "legit_slug_coverage": {s: slug_counts.get(s, 0) for s in LEGIT_SLUGS},
        "n_legit_slugs_covered": sum(1 for s in LEGIT_SLUGS if slug_counts.get(s)),
        "legit_profile_entropy_bits": round(_entropy(slug_counts), 3),
        "steering_uptake_mean": (sum(uptake) / len(uptake)) if uptake else None,
        "val_game_ids": sorted(val_ids),
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "sft_train.parquet")
    val_path = out_dir / "sft_val.parquet"
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(val_path)
    elif val_path.exists():
        # A stale val parquet would overlap train, and the trainer's default VAL_FILES reads it.
        val_path.unlink()
        print(f"[build_sft] --val-games 0: removed stale {val_path.name} (all rows are train now)")
    (out_dir / "sft_build_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"[build_sft] {len(train_rows)} train / {len(val_rows)} val rows "
          f"({len(rows)} supervised turns; too_long={n_too_long}, dedup={n_dedup}) -> {out_dir}")
    print(f"[build_sft] clean-prefix hist: {dict(sorted(prefix_hist.items()))}; "
          f"truncations: {dict(truncate_reasons.most_common())}")
    print(f"[build_sft] legit coverage: {stats['n_legit_slugs_covered']}/{len(LEGIT_SLUGS)} slugs, "
          f"entropy {stats['legit_profile_entropy_bits']} bits, "
          f"steering uptake {stats['steering_uptake_mean']}")


if __name__ == "__main__":
    main()
