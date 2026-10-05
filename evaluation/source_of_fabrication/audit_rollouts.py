#!/usr/bin/env python3
"""Stage A2: extract each game's fabricated claims from a rl_rollout.py result file.

Calls the fabrication judge exactly as evaluation/audit_fabrications.py does for old-bailey
(`rl.monitors._audit_evidence` over the per-round <argument> text, the annotated full-template
evidence via `rl.monitors._labeled_evidence`, `game_background` as CASE BACKGROUND), but keeps the
claim strings, one line per game, for splicing back into the conversation. The judge does not
report rounds, so `round_judge` is null and build_probe_items.attribute_round recovers the round.

Judge: the served model via rl.receiver_client (RECEIVER_HOST / RECEIVER_PORT / RECEIVER_MODEL_ID).
Only `complete` games are audited. Resume skips audited game_ids and retries judge_fail games; the
file is append-only and readers keep the latest record per game.

  RECEIVER_HOST=127.0.0.1 RECEIVER_PORT=20001 RECEIVER_MODEL_ID=qwen3.5-35B \\
    python evaluation/source_of_fabrication/audit_rollouts.py \\
        --result-file $SOF_ROOT/rollouts/rollout__8B__gs100.json \\
        --out $SOF_ROOT/audits/audit__8B__gs100.jsonl
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.audit_fabrications import collect_sender_args  # noqa: E402
from rl import receiver_client  # noqa: E402  (current_game_id: judge-dump attribution)
from rl.monitors import _audit_evidence, _labeled_evidence  # noqa: E402

# The annotated Old Bailey FULL template the judge is shown as the EVIDENCE SET.
TEMPLATE_REL = "datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json"


def default_template() -> str:
    return str(_REPO / TEMPLATE_REL)


def annotated_by_id(template_path):
    """game id -> the annotated evidence list the judge is shown as the EVIDENCE SET."""
    with open(template_path) as fh:
        games = json.load(fh)
    return {g["id"]: (g.get("params", {}).get("private", {}).get("information") or [])
            for g in games}


def audit_game(game: dict, annotated: dict) -> dict:
    """One game's audit record. Sets receiver_client.current_game_id on the pool thread so recorded
    judge calls are attributed to this game."""
    gid = game.get("id")
    token = receiver_client.current_game_id.set(gid)
    try:
        arguments = collect_sender_args(game)
        info = annotated.get(gid)
        if info is None:
            return {"game_id": gid, "judge_fail": True,
                    "error": "game id not in the annotated template"}
        if not arguments:
            return {"game_id": gid, "judge_fail": False, "arguments": arguments, "genuine": [],
                    "raw_n": 0, "set_size": 0, "coverage_idx": []}
        background = ((game.get("params") or {}).get("public") or {}).get("game_background")
        res = _audit_evidence(arguments, _labeled_evidence(info), background=background)
        if res is None:
            return {"game_id": gid, "judge_fail": True, "arguments": arguments}
        return {"game_id": gid, "judge_fail": False, "arguments": arguments,
                # round_judge is null: the judge does not report rounds
                "genuine": [{"claim": c, "round_judge": None} for c in res["genuine"]],
                "raw_n": res["raw_n"], "set_size": res["set_size"],
                "coverage_idx": sorted(res["coverage_idx"])}
    finally:
        receiver_client.current_game_id.reset(token)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--result-file", required=True, help="result JSON from evaluation/rl_rollout.py")
    ap.add_argument("--out", required=True, help="audit JSONL (appended; audited game_ids skipped)")
    ap.add_argument("--full-template", default=default_template(),
                    help="annotated FULL template (the judge's evidence ground truth)")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap games audited (0 = all)")
    args = ap.parse_args()

    annotated = annotated_by_id(args.full_template)
    games = [g for g in json.load(open(args.result_file)) if g.get("complete")]
    # Resume: skip only successful audits. judge_fail records (e.g. after a judge-server death;
    # receiver_client.chat returns "" without raising) are retried.
    done = set()
    n_fail_seen = 0
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("judge_fail"):
                    n_fail_seen += 1
                else:
                    done.add(rec["game_id"])
        print(f"[audit] resume: {len(done)} games already audited "
              f"({n_fail_seen} earlier judge failures will be retried)")
    todo = [g for g in games if g["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[audit] {len(games)} complete games, {len(todo)} to audit")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_fab = n_fail = 0
    with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(audit_game, g, annotated) for g in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_fab += len(rec.get("genuine") or [])
            n_fail += bool(rec.get("judge_fail"))
            if i % 20 == 0 or i == len(futures):
                print(f"  ... {i}/{len(futures)} audited ({n_fab} false claims, "
                      f"{n_fail} judge failures)", flush=True)

    by_gid, total = {}, 0
    with open(args.out) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_gid[rec["game_id"]] = rec       # latest record per game wins
    still_failed = sum(bool(r.get("judge_fail")) for r in by_gid.values())
    for rec in by_gid.values():
        total += len(rec.get("genuine") or [])
    print(f"[audit] done -> {args.out}  ({len(by_gid)} games, {total} false claims, "
          f"{still_failed} still judge-failed)")


if __name__ == "__main__":
    main()
