#!/usr/bin/env python3
"""Enumerate the cells of the reward-landscape grid and write the manifest the audit job reads.

Released grid: 44 prompt arms (base, the 42-technique guide, one `single_strategy:<slug>` per
technique) x 2 senders x the stubborn juror = 88 cells. --senders / --jurors / --prompts change it.

Paths are built with `audit_source.result_path` and must exist; never globbed, because a sender dir
also holds other studies' cells with the same `*_oldbailey_rlrollout_val_sender-*` shape.

Writes a TSV with a header under `<results root>/reward_landscape/`:

    key      <sender dir>__<profile>__<arm>, the name of the cell's audit artifacts
    sender   results-directory slug (the audit job passes it as --model)
    profile  juror profile
    arm      base | strat | only_<slug>
    result   the cell's rollout transcript

  python evaluation/reward_landscape/build_manifest.py [--results-root DIR] [--out FILE]
  python evaluation/reward_landscape/build_manifest.py --selftest
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.reward_landscape import audit_source  # noqa: E402
from rl.sender_prompts import all_specs, arm_key, parse_spec  # noqa: E402

# The two senders of the paper's panel, by results-directory slug.
PAPER_SENDERS = ("qwen3-4B-instruct-base", "qwen3-8B-base")
PAPER_JURORS = ("stubborn",)
COLUMNS = ("key", "sender", "profile", "arm", "result")


def cells(senders, jurors, specs, results_root=None):
    """One row per (sender, juror profile, prompt arm), in a stable order."""
    for sender_dir in senders:
        for profile in jurors:
            for spec in specs:
                yield {
                    "key": f"{sender_dir}__{profile}__{arm_key(spec)}",
                    "sender": sender_dir,
                    "profile": profile,
                    "arm": arm_key(spec),
                    "result": audit_source.result_path(sender_dir, profile, spec, results_root),
                }


def selftest() -> int:
    specs = all_specs(audit_source.DOMAIN)
    assert len(specs) == 44, f"expected 44 prompt arms, got {len(specs)}"
    rows = list(cells(PAPER_SENDERS, PAPER_JURORS, specs, "/results"))
    assert len(rows) == len(PAPER_SENDERS) * len(PAPER_JURORS) * 44, "grid size"
    assert len({r["key"] for r in rows}) == len(rows), "duplicate cell key"
    for r in rows:
        assert audit_source.key_for_result(r["result"]) == r["key"], \
            f"{r['result']} does not reverse to {r['key']}"
        assert ":" not in r["key"], f"{r['key']}: an arm key carries ':'"
    print(f"[manifest] selftest OK: {len(specs)} arms, {len(rows)} unique keys, "
          f"every result path reverses to its key")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--results-root", default=None,
                    help="root of every input and output (default: $RESULTS_ROOT)")
    ap.add_argument("--senders", nargs="+", default=list(PAPER_SENDERS),
                    help="sender results-directory slugs (default: the two paper senders)")
    ap.add_argument("--jurors", nargs="+", default=list(PAPER_JURORS),
                    help="juror profiles (default: stubborn)")
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="sender prompt specs (default: all 44 -- base, strategies and one "
                         "single_strategy:<slug> per technique)")
    ap.add_argument("--out", default=None,
                    help="manifest path (default: <results root>/reward_landscape/manifest.tsv)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the arm alphabet and the key round-trip; touches no results")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    results_root = a.results_root or audit_source.default_results_root()
    specs = ([parse_spec(p) for p in a.prompts] if a.prompts
             else all_specs(audit_source.DOMAIN))
    out = Path(a.out or Path(audit_source.default_out_dir(results_root)) / "manifest.tsv")

    rows = list(cells(a.senders, a.jurors, specs, results_root))
    assert len({r["key"] for r in rows}) == len(rows), "duplicate cell key"

    missing = [r for r in rows if not Path(r["result"]).is_file()]
    if missing:
        print(f"[manifest] ERROR: {len(missing)}/{len(rows)} result transcripts missing, e.g.:",
              file=sys.stderr)
        for r in missing[:5]:
            print(f"    {r['result']}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in COLUMNS) + "\n")

    print(f"[manifest] {len(rows)} cells -> {out}")
    print(f"[manifest] senders: {' '.join(a.senders)}")
    print(f"[manifest] jurors: {' '.join(a.jurors)}")
    print(f"[manifest] arms per sender x juror: {len(specs)}")
    print("[manifest] all result transcripts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
