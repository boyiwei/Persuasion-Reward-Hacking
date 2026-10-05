#!/usr/bin/env python3
"""Gate the audit artifacts before any number is computed from them.

The figure assumes every cell was judged by one judge instance under the current judge code; a
resumed job, stale checkout or half-finished pass breaks that silently. Checks, in failure order:
  1. `_judge_instance.txt` is present (else provenance is unverifiable)
  2. every manifest cell has both a strategy audit and a fabrication sidecar
  3. every strategy audit has the provenance keys with expected values (1024/256 judge budgets,
     false_information preset from the fabrication count)
  4. every fabrication sidecar has evidence_audit_retry (without the parse retry claim-rich games
     drop out, so cells are not comparable)
  5. one judge model across the directory, the one the instance file names
  6. every audit is taxonomy-wide, and audited + no-argument games cover the whole transcript
     (no silent --n-games subsample)
  7. the producing job left an empty FAILED.tsv

  python evaluation/reward_landscape/check_inputs.py [--audit-dir DIR] [--manifest FILE]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.reward_landscape import audit_source  # noqa: E402
from rl.strategy_audit.taxonomy import slugs  # noqa: E402

# Below this many audited games, warn but don't fail: a refusal-heavy arm is a result, not a fault.
MIN_AUDITED = 70
# Code-version contracts, not properties of a particular results tree.
EXPECT_FI_SOURCE = "genuine_fabrication_count>=1"
EXPECT_BUDGET = {"grounded": 1024, "ungrounded": 256}


def read_manifest(p: Path):
    lines = p.read_text().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def judge_of_instance(text: str):
    """The `judge=` field of an `_judge_instance.txt` line (by key: `path=` also has the name)."""
    for tok in text.split():
        if tok.startswith("judge="):
            return tok[len("judge="):]
    return None


def n_games_in(result: str):
    """Games in a cell's rollout transcript -- the number its audit has to account for."""
    with open(result) as fh:
        return len(json.load(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--results-root", default=None,
                    help="root of every input and output (default: $RESULTS_ROOT)")
    ap.add_argument("--audit-dir", default=None,
                    help="the audit job's by_result/ directory (default: $AUDIT_DIR)")
    ap.add_argument("--manifest", default=os.getenv("MANIFEST"),
                    help="the cell list to gate (default: $MANIFEST, else "
                         "<results root>/reward_landscape/manifest.tsv)")
    a = ap.parse_args(argv)
    results_root = a.results_root or audit_source.default_results_root()
    AD = Path(a.audit_dir or audit_source.default_audit_dir(results_root))
    manifest = Path(a.manifest or Path(audit_source.default_out_dir(results_root))
                    / "manifest.tsv")
    if not manifest.is_file():
        print(f"[check] FAIL: no manifest at {manifest} -- run build_manifest.py")
        return 1
    rows = read_manifest(manifest)
    print(f"[check] manifest {len(rows)} cells; audit dir {AD}")
    if not AD.is_dir():
        print(f"[check] FAIL: audit dir does not exist: {AD}")
        print("        run the audit, or point --audit-dir / $AUDIT_DIR at its j<jobid>/by_result")
        return 1

    fails, warns = [], []
    expect_strategies = len(slugs())

    # 1: the instance file names the judge every artifact must match
    inst = AD / "_judge_instance.txt"
    if not inst.is_file():
        print(f"[check] FAIL: no _judge_instance.txt in {AD} -- provenance unverifiable")
        return 1
    instance = inst.read_text().strip()
    expect_judge = judge_of_instance(instance)
    print(f"[check] judge instance: {instance}")
    if not expect_judge:
        fails.append(f"{inst} carries no judge= field")

    # 2: presence
    missing_sa = [r["key"] for r in rows if not (AD / f"{r['key']}.strategy_audit.json").is_file()]
    missing_fb = [r["key"] for r in rows if not (AD / f"{r['key']}.fabrication.json").is_file()]
    if missing_sa:
        fails.append(f"{len(missing_sa)}/{len(rows)} strategy audits missing (e.g. {missing_sa[:3]})")
    if missing_fb:
        fails.append(f"{len(missing_fb)}/{len(rows)} fabrication sidecars missing (e.g. {missing_fb[:3]})")

    judges, fi_sources, budgets, retries, n_games = Counter(), Counter(), Counter(), Counter(), Counter()
    bad_keys, bad_nstrat, thin = [], [], []
    expect_games = {}
    for r in rows:
        p = AD / f"{r['key']}.strategy_audit.json"
        if not p.is_file():
            continue
        d = json.load(open(p))
        miss = [k for k in audit_source.POSTFIX_KEYS if k not in d]
        if miss:
            bad_keys.append((r["key"], miss))
        judges[d.get("judge_model")] += 1
        fi_sources[d.get("false_information_source")] += 1
        budgets[json.dumps(d.get("judge_budget_policy"), sort_keys=True)] += 1
        # No-argument games are not audited, so the invariant is audited + no-argument == the
        # transcript's games, not n_games_audited == n_games.
        expect_games[r["key"]] = n_games_in(r["result"])
        n_aud, n_noarg = d.get("n_games_audited"), d.get("n_games_no_argument", 0)
        if n_aud is None or n_aud + n_noarg != expect_games[r["key"]]:
            n_games[f"{n_aud}+{n_noarg} of {expect_games[r['key']]}"] += 1
        if n_aud is not None and n_aud < MIN_AUDITED:
            thin.append((r["key"], n_aud, n_noarg))
        if d.get("n_strategies") != expect_strategies:
            bad_nstrat.append((r["key"], d.get("n_strategies")))

        f = AD / f"{r['key']}.fabrication.json"
        if f.is_file():
            m = json.load(open(f)).get("metrics", {})
            retries[m.get("evidence_audit_retry", "<absent>")] += 1

    if bad_keys:
        fails.append(f"{len(bad_keys)} strategy audits missing provenance keys (e.g. {bad_keys[:2]})")
    if bad_nstrat:
        fails.append(f"{len(bad_nstrat)} audits not {expect_strategies}-way (e.g. {bad_nstrat[:2]})")
    if expect_judge and set(judges) - {expect_judge}:
        fails.append(f"judge_model not uniformly {expect_judge}: {dict(judges)}")
    if set(fi_sources) - {EXPECT_FI_SOURCE}:
        fails.append(f"false_information_source not uniformly {EXPECT_FI_SOURCE!r}: {dict(fi_sources)}")
    if set(budgets) - {json.dumps(EXPECT_BUDGET, sort_keys=True)}:
        fails.append(f"judge_budget_policy not uniformly {EXPECT_BUDGET}: {dict(budgets)}")
    if n_games:
        fails.append("cells where audited+no_argument != the games in the transcript "
                     f"(games unaccounted for): {dict(n_games)}")
    if thin:
        warns.append(f"{len(thin)} cells audited < {MIN_AUDITED} games (sender refusals): {thin}")
    if "<absent>" in retries:
        fails.append(f"{retries['<absent>']} fabrication sidecars lack evidence_audit_retry")
    elif len(retries) > 1:
        warns.append(f"mixed evidence_audit_retry values: {dict(retries)}")

    # 7: the producing job's own failure log
    failed = AD.parent / "FAILED.tsv"
    if failed.is_file() and failed.stat().st_size > 0:
        fails.append(f"producing job recorded failures in {failed}:\n"
                     + "\n".join("    " + ln for ln in failed.read_text().splitlines()[:10]))

    for w in warns:
        print(f"[check] WARN: {w}")
    if fails:
        print(f"[check] FAIL ({len(fails)}):")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    games = sorted(set(expect_games.values()))
    print(f"[check] OK: {len(rows)}/{len(rows)} cells, {expect_strategies}-way, "
          f"{games[0] if len(games) == 1 else games} games per cell, one judge "
          f"({expect_judge}), current provenance on every artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
