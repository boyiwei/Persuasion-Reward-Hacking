#!/usr/bin/env python3
"""Stage C: ask one served policy whether each statement is true, in or out of context.

The prober is a local OpenAI-compatible SGLang endpoint answering every item under one
(context-mode, arm, assertion) cell; prompt text comes from probe_prompts.py.

  * No greedy readout: answers use the policy's standard sampling (temp 1 / top_p 1 / top_k -1,
    the RL-rollout setting). Draw #1 is the readout by position; draws #2..N measure stability.
  * Every answer row records who answered (served_name) and which checkpoint produced the evidence
    (origin_ckpt / origin_step); the analyzer keys on both.
  * Resume keys on readout_response; __ERROR__ rows are retried.

  python evaluation/source_of_fabrication/run_probe.py --port 20000 --served-name qwen3-8B-gs100 \\
      --items $SOF_ROOT/probe_items_union/union__8B.jsonl \\
      --contexts $SOF_ROOT/probe_items_union/contexts_union__8B.jsonl.gz \\
      --context-mode in --arm role --assertion pre \\
      --out $SOF_ROOT/answers_union/8B__Pgs100__in_pre_role.jsonl
"""
import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from openai import OpenAI  # noqa: E402

from evaluation.source_of_fabrication.probe_prompts import compose  # noqa: E402

# Only these keys reach the answer row; without origin_ckpt/origin_step the origin axis is lost.
_ITEM_KEYS = ("size", "design", "step", "ckpt", "origin_ckpt", "origin_step", "kind", "game_id",
              "item_id", "source_claim_type", "statement", "round", "round_judge", "match_method",
              "context_id")

# The policy's standard generation setting (matches evaluation/rl_rollout.py / the GRPO rollout).
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--items", nargs="+", required=True, help="probe-item JSONL file(s)")
    ap.add_argument("--contexts", default=None, help="context table (.jsonl.gz); required for in-context")
    ap.add_argument("--context-mode", choices=["in", "out"], required=True)
    ap.add_argument("--arm", choices=["honest", "role"], default="honest")
    ap.add_argument("--assertion", choices=["pre", "post"], default="pre")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=3,
                    help="draws per item at standard sampling; #1 = readout, #2.. = stability")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap total items (0 = all)")
    args = ap.parse_args()
    if args.n_draws < 1:
        ap.error("--n-draws must be >= 1 (draw #1 is the classified readout)")

    items = []
    for f in args.items:
        if not os.path.exists(f):
            print(f"[probe] WARN missing {f}")
            continue
        with open(f) as fh:
            items.extend(json.loads(line) for line in fh if line.strip())
    if args.limit:
        # Plain head-slice: the union is sorted by origin, so this probes only base-origin items.
        # Use probe_items_union/union_smoke__*.jsonl for smokes.
        items = items[:args.limit]

    # Every item needs both provenance labels, or the 3x3 design degenerates.
    bad = [it["item_id"] for it in items
           if it.get("origin_ckpt") is None or it.get("origin_step") is None
           or it["origin_ckpt"] != it.get("ckpt") or it["origin_step"] != it.get("step")]
    if bad:
        raise SystemExit(f"[probe] {len(bad)} items lack a consistent origin label e.g. {bad[:3]} "
                         f"— rebuild with build_item_set.py")

    contexts = {}
    if args.context_mode == "in":
        if not args.contexts:
            ap.error("--contexts is required with --context-mode in")
        with gzip.open(args.contexts, "rt") as fh:
            for line in fh:
                if line.strip():
                    c = json.loads(line)
                    contexts[c["context_id"]] = c
        missing = {it["context_id"] for it in items} - set(contexts)
        if missing:
            raise SystemExit(f"[probe] {len(missing)} item context_ids missing from {args.contexts}")

    # Resume: skip only non-error readouts. __ERROR__ rows (e.g. the server died) are retried and
    # append a second row; classify keeps the latest per item_id.
    done = set()
    n_err = 0
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if str(rec.get("readout_response", "")).startswith("__ERROR__"):
                    n_err += 1
                else:
                    done.add(rec["item_id"])
        print(f"[probe] resume: {len(done)} answers already in {args.out} "
              f"({n_err} earlier __ERROR__ rows will be retried)")
    todo = [it for it in items if it["item_id"] not in done]
    tag = args.context_mode if args.context_mode == "out" else f"in/{args.assertion}/{args.arm}"
    print(f"[probe] {args.served_name} [{tag}]: {len(todo)}/{len(items)} items to run "
          f"({args.n_draws} draws each at temp={SAMPLING['temperature']})")

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="None")
    # Native <think> off; every draw uses the same standard sampling.
    base_extra = {"chat_template_kwargs": {"enable_thinking": False},
                  "min_p": 0.0, "repetition_penalty": 1.0}

    def call(messages, temperature, top_p, top_k, attempts=4):
        last = ""
        for _ in range(attempts):
            try:
                r = client.chat.completions.create(
                    model=args.served_name, messages=messages, temperature=temperature, top_p=top_p,
                    presence_penalty=0.0, max_tokens=args.max_tokens,
                    extra_body={**base_extra, "top_k": top_k})
                return (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001  - retry transient server hiccups
                last = str(e)
        return f"__ERROR__ {last}"

    def work(item):
        messages = compose(item, contexts, args.context_mode, args.arm, args.assertion)
        draws = [call(messages, SAMPLING["temperature"], SAMPLING["top_p"], SAMPLING["top_k"])
                 for _ in range(args.n_draws)]
        out = {k: item.get(k) for k in _ITEM_KEYS}
        out.update(served_name=args.served_name, context_mode=args.context_mode,
                   arm=args.arm if args.context_mode == "in" else None,
                   assertion=args.assertion if args.context_mode == "in" else None,
                   readout_response=draws[0], samples=draws[1:], sampling=dict(SAMPLING))
        return out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for res in ex.map(work, todo):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 100 == 0:
                print(f"  ... {n}/{len(todo)}", flush=True)
    print(f"[probe] wrote {n} answers -> {args.out}")


if __name__ == "__main__":
    main()
