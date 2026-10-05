#!/usr/bin/env python3
"""Gate one audit artifact: exit 0 iff it is complete and was produced by the pinned judge.

1. `evaluation.rollout_integrity.check_audit` as a library: a judge that dies mid-audit still writes
   a complete-looking file with rates on a reduced denominator. Its CLI cannot override
   `n_strategies` (so the 2% gate loosens on narrower artifacts); the artifact's own is passed.
2. Provenance: judge-budget policy, false_information source, judge name, and coverage against a
   count from the cell's own transcript. The fabrication audit scores every game; the strategy
   audit drops no-argument games, then subsamples to --n-games, so it covers either every game
   (audited + no-argument, the figure's case) or exactly --n-games audited.

  python evaluation/reward_landscape/gate_audit.py --strategy <f.strategy_audit.json> \
      --result <rollout.json> [--n-games N]
  python evaluation/reward_landscape/gate_audit.py --fabrication <f.fabrication.json> \
      --result <rollout.json>
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.rollout_integrity import check_audit  # noqa: E402
from rl.strategy_audit.taxonomy import slugs  # noqa: E402

EXPECT_BUDGET = {"grounded": 1024, "ungrounded": 256}
EXPECT_FI_SOURCE = "genuine_fabrication_count>=1"
DEFAULT_JUDGE = "qwen3.5-35B"


def games_in_result(path: str) -> int:
    with open(path) as fh:
        return len(json.load(fh))


def gate_strategy(path: str, expect_strategies: int, max_fail_frac: float,
                  n_result: int, n_games, expect_judge: str) -> int:
    try:
        d = json.loads(Path(path).read_text())
    except Exception as e:                                        # noqa: BLE001
        print(f"[gate] unreadable: {path} ({e})")
        return 1
    bad = []
    n_str = d.get("n_strategies")
    if expect_strategies and n_str != expect_strategies:
        bad.append(f"n_strategies={n_str}, expected {expect_strategies}")

    # provenance: the current judge code stamps all three
    if d.get("judge_budget_policy") != EXPECT_BUDGET:
        bad.append(f"judge_budget_policy={d.get('judge_budget_policy')} != {EXPECT_BUDGET}")
    if d.get("false_information_source") != EXPECT_FI_SOURCE:
        bad.append(f"false_information_source={d.get('false_information_source')!r} "
                   f"!= {EXPECT_FI_SOURCE!r}")
    grounded = d.get("judge_budget_grounded_slugs")
    if not isinstance(grounded, list):
        bad.append("judge_budget_grounded_slugs missing")
    elif "false_information" in grounded:
        # must be preset from the fabrication count, never judged
        bad.append("false_information appears in judge_budget_grounded_slugs -- it was judged")
    if d.get("judge_model") != expect_judge:
        bad.append(f"judge_model={d.get('judge_model')!r} != {expect_judge!r}")

    # coverage: complete = the whole result (audited + no-argument) or, when --n-games
    # subsamples, exactly that many audited games
    n_aud = d.get("n_games_audited") or 0
    n_noarg = d.get("n_games_no_argument") or 0
    if n_aud + n_noarg != n_result and not (n_games and n_aud == n_games):
        want = f"{n_result}" if not n_games else f"{n_result} (or {n_games} audited)"
        bad.append(f"n_games_audited+n_games_no_argument={n_aud}+{n_noarg} != {want}")

    if bad:
        print(f"[gate] REJECT {Path(path).name}: " + "; ".join(bad))
        return 1
    # Layer 1 last, with the artifact's own slug count as denominator.
    if check_audit(path, max_fail_frac, n_strategies=n_str) != 0:
        return 1
    print(f"[gate] ok {Path(path).name} (n_strategies={n_str}, "
          f"parse_fail={d.get('n_parse_fail_calls')})")
    return 0


def gate_fabrication(path: str, expect_games: int, expect_judge: str) -> int:
    try:
        d = json.loads(Path(path).read_text())
    except Exception as e:                                        # noqa: BLE001
        print(f"[gate] unreadable: {path} ({e})")
        return 1
    m = d.get("metrics") or {}
    rows = d.get("rows") or []
    bad = []
    # Stamped by audit_fabrications.py's parse retry; a one-shot audit drops claim-rich games.
    if "evidence_audit_retry" not in m:
        bad.append("metrics.evidence_audit_retry missing")
    if len(rows) != expect_games:
        bad.append(f"{len(rows)} rows, expected {expect_games}")
    if m.get("judge_model") != expect_judge:
        bad.append(f"judge_model={m.get('judge_model')!r} != {expect_judge!r}")
    missing = [r.get("id") for r in rows
               if not r.get("judge_fail") and r.get("rh_fake_evidence") is None]
    if missing:
        bad.append(f"{len(missing)} non-judge_fail rows have no rh_fake_evidence")
    if bad:
        print(f"[gate] REJECT {Path(path).name}: " + "; ".join(bad))
        return 1
    jf = m.get("n_game_judge_fail", 0)
    print(f"[gate] ok {Path(path).name} (judge_fail={jf}/{len(rows)}, "
          f"retry={m.get('evidence_audit_retry')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--strategy", help="path to a .strategy_audit.json")
    ap.add_argument("--fabrication", help="path to a .fabrication.json")
    ap.add_argument("--result", required=True,
                    help="the cell's rollout transcript: the games the audit had to score")
    ap.add_argument("--n-games", type=int, default=None,
                    help="the --n-games the strategy audit was run with, when it subsamples")
    ap.add_argument("--expect-judge", default=os.getenv("JUDGE_NAME") or DEFAULT_JUDGE)
    ap.add_argument("--max-fail-frac", type=float, default=0.02)
    a = ap.parse_args(argv)
    if bool(a.strategy) == bool(a.fabrication):
        ap.error("pass exactly one of --strategy / --fabrication")
    try:
        n_result = games_in_result(a.result)
    except Exception as e:                                        # noqa: BLE001
        print(f"[gate] unreadable result: {a.result} ({e})")
        return 1
    if a.strategy:
        return gate_strategy(a.strategy, len(slugs()), a.max_fail_frac, n_result, a.n_games,
                             a.expect_judge)
    # The fabrication audit has no game-count flag: it scores every game in the result.
    return gate_fabrication(a.fabrication, n_result, a.expect_judge)


if __name__ == "__main__":
    sys.exit(main())
