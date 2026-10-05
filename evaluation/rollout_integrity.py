#!/usr/bin/env python
"""Integrity check / repair for an rl_rollout result JSON, so a replicate survives a server that
dies mid-run (a short allocation guarantees this).

rl_rollout.py marks a game complete from the sender alone (`out["complete"] = not sender_failed`),
and rl.receiver_client.chat returns '' after its retries instead of raising. If the served 35B dies
mid-run, the remaining games are written complete with an empty juror response, `--skip` resume
never replays them, and they enter the metrics as belief-extraction failures.

Damage means an empty string a live model can't produce: a complete game with fewer `responses`
than expected rounds, or any round with empty `receiver` or `sender` text. A receiver that answered
with an unparseable belief is real behaviour, not damage.

The strategy audit has the same failure mode: rl.strategy_audit.run records a failed judge call as
None and computes rates over what answered, so a judge dying mid-audit silently shrinks the
denominator. --check-audit rejects such a file so the cell is re-judged.

Modes:
  --check  <res>          exit 0 iff every game is complete and undamaged (for the resume test)
  --repair <res>          demote damaged games to complete=false and rewrite the file; exit 0 iff
                          the file is then fully complete, 1 if any game still needs playing.
  --check-audit <audit>   exit 0 iff the audit is trustworthy: judge-failure rate under
                          --max-fail-frac and every game judged for the illegal rollup.
"""
import argparse
import json
import os
import sys


def _empty(s):
    return not (s or "").strip()


def damaged_rounds(game, rounds_expected):
    """Indices of rounds in this game that carry a dead-server signature."""
    resp = game.get("responses") or []
    bad = [i for i, r in enumerate(resp)
           if _empty(r.get("receiver")) or _empty(r.get("sender"))]
    if len(resp) < rounds_expected:
        bad.append(-1)          # short game: rounds are missing outright
    return bad


def scan(games, rounds_expected):
    damaged, pending = [], []
    for g in games:
        if not g.get("complete"):
            pending.append(g.get("id"))
            continue
        if damaged_rounds(g, rounds_expected):
            damaged.append(g.get("id"))
    return damaged, pending


def check_audit(path, max_fail_frac, n_strategies=42):
    """Exit-code helper: is this .strategy_audit.json trustworthy? See the module docstring."""
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        print(f"[integrity] audit unreadable ({e})")
        return 1
    n_games = d.get("n_games_audited") or 0
    if not n_games:
        print("[integrity] audit has no games")
        return 1
    fails = d.get("n_parse_fail_calls") or 0
    total = n_strategies * n_games
    frac = fails / total if total else 1.0
    judged = d.get("n_games_illegal_judged")
    bad = []
    if frac > max_fail_frac:
        bad.append(f"judge-failure rate {frac:.1%} ({fails}/{total}) > {max_fail_frac:.1%}")
    if judged is not None and judged < n_games:
        bad.append(f"illegal rollup covers {judged}/{n_games} games")
    if bad:
        print(f"[integrity] UNTRUSTWORTHY AUDIT {os.path.basename(path)}: " + "; ".join(bad))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("path")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 0 iff the file is complete and undamaged")
    ap.add_argument("--repair", action="store_true",
                    help="demote damaged games to complete=false and rewrite")
    ap.add_argument("--check-audit", action="store_true",
                    help="validate a .strategy_audit.json instead of a rollout")
    ap.add_argument("--max-fail-frac", type=float, default=0.02,
                    help="max tolerated fraction of failed judge calls (default 0.02)")
    ap.add_argument("--rounds", type=int, default=3, help="expected rounds per game (default 3)")
    a = ap.parse_args()
    if a.check_audit:
        return check_audit(a.path, a.max_fail_frac)
    if a.check == a.repair:
        ap.error("pass exactly one of --check / --repair / --check-audit")

    try:
        games = json.load(open(a.path))
    except Exception as e:  # noqa: BLE001 - a truncated checkpoint is a normal resume state
        print(f"[integrity] unreadable ({e}) -> treat as incomplete")
        return 1

    damaged, pending = scan(games, a.rounds)

    if a.check:
        if damaged or pending:
            print(f"[integrity] {a.path}: {len(damaged)} damaged, {len(pending)} unplayed "
                  f"(of {len(games)})")
            return 1
        return 0

    if damaged:
        dset = set(damaged)
        for g in games:
            if g.get("id") in dset:
                g["complete"] = False
        with open(a.path + ".tmp", "w") as fh:
            json.dump(games, fh, indent=4)
        os.replace(a.path + ".tmp", a.path)
        print(f"[integrity] {a.path}: demoted {len(damaged)} game(s) with an empty sender/receiver "
              f"turn back to incomplete (dead-server signature) — they will be replayed")
    n_ok = sum(1 for g in games if g.get("complete"))
    print(f"[integrity] {a.path}: {n_ok}/{len(games)} complete after repair")
    return 0 if n_ok == len(games) else 1


if __name__ == "__main__":
    sys.exit(main())
