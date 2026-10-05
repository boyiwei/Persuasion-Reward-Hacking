#!/usr/bin/env python3
"""Render the exact judge prompts the audits send for one real game, without a judge (no GPU).

Replaces only the transport (rl.receiver_client.chat) with a recorder, drives the shipped paths
(audit.classify for the grounded techniques, monitors.compute_monitors for the evidence audit) and
prints each prompt with its call_type and max_tokens. The evidence audit uses the result file's
cleaned evidence rather than the annotated template or parquet payload; the judge sees only raw
descriptions, so the text matches the wire for every RL-parity result.

  .venv/bin/python -m rl.strategy_audit.render_check --game-id 18 \\
      --result experiments/results/old-bailey/<...>/stubborn_oldbailey_rlrollout_val__recv_qwen3.5-35B_rep4.json
"""
import argparse
import sys
from pathlib import Path

from rl import monitors, receiver_client
from rl.strategy_audit import audit
from rl.strategy_audit.run import _load_games_result
from rl.strategy_audit.taxonomy import by_slug


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True, help="result JSON (evaluation/rl_rollout.py schema)")
    p.add_argument("--game-id", type=int, required=True, help="the game's `id` field")
    p.add_argument("--slugs", default=",".join(audit._GROUNDED),
                   help="comma-separated strategy slugs to render (default: the grounded ones)")
    p.add_argument("--domain", default=audit.DEFAULT_DOMAIN, choices=audit.DOMAINS)
    p.add_argument("--no-fabrication", action="store_true", help="skip the evidence-audit prompt")
    a = p.parse_args(argv)

    games, _ = _load_games_result(Path(a.result), 0, 0)
    g = next((x for x in games if x.get("id") == a.game_id), None)
    if g is None:
        sys.exit(f"[render_check] no game with id {a.game_id} in {a.result}")
    argument_text = "\n\n".join(g["arguments"])  # the same join rl.strategy_audit.run uses

    captured = []

    def _fake_chat(messages, max_tokens=None, **_kw):
        captured.append({"messages": messages, "max_tokens": max_tokens})
        return '{"items_used": [], "false_claims": []}' if "false_claims" in messages[0]["content"] \
            else "<answer>0</answer>"

    real = receiver_client.chat
    receiver_client.chat = _fake_chat
    try:
        reg = by_slug(a.domain)
        for slug in [s.strip() for s in a.slugs.split(",") if s.strip()]:
            captured.clear()
            audit.classify(reg[slug], argument_text, None, 0, g["evidence"], g.get("background"),
                           domain=a.domain)
            c = captured[-1]
            print(f"\n{'=' * 100}\n[strategy judge] slug={slug} call=strategy_{slug} "
                  f"max_tokens={c['max_tokens']}\n{'=' * 100}")
            print(c["messages"][0]["content"])
        if not a.no_fabrication:
            captured.clear()
            monitors.drain_judge_records()
            monitors.compute_monitors(g["arguments"], g["evidence"], background=g.get("background"))
            recs = monitors.drain_judge_records()
            for r in recs:
                print(f"\n{'=' * 100}\n[{r['call_type']}] max_tokens={r['max_tokens']}\n{'=' * 100}")
                print(r["messages"][0]["content"])
    finally:
        receiver_client.chat = real
    print(f"\n[render_check] game {a.game_id}: {len(g['arguments'])} rounds, {len(g['evidence'])} evidence items, "
          f"background={'yes' if g.get('background') else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
