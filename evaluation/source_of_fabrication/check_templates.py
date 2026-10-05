#!/usr/bin/env python3
"""Validity gate: the probe checkpoints of one size must share one chat template.

A merged checkpoint with a different `chat_template` would change the prompt for that prober alone,
indistinguishable from a real respondent effect. Base HF models embed the template in
`tokenizer_config.json["chat_template"]`, `verl.model_merger` writes `chat_template.jinja`;
`effective_template` reads either. `ckpt_path` resolves (size, ckpt) like ckpt_paths.sh, with the
same MODELS_DIR / TC / MERGED / PROFILE overrides.

Modes:
  * preflight (several --ckpts): exit non-zero unless every prober of a size has the same template.
  * record (--path, one size and ckpt): hash the directory a job is about to serve and write
    (source, len, sha256_16) next to its answers; `--verify-dir` later applies the identity gate
    to those records.

  python evaluation/source_of_fabrication/check_templates.py --sizes 4B 8B --out tpl.json
  python evaluation/source_of_fabrication/check_templates.py --sizes 8B --ckpts gs100 \\
      --path <ckpt dir> --out $ANS_DIR/template_check_8B_gs100.json
  python evaluation/source_of_fabrication/check_templates.py --verify-dir $ANS_DIR --sizes 4B 8B
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODELS_DIR = os.environ.get("MODELS_DIR") or str(_REPO / "models")
TC = os.environ.get("TC", MODELS_DIR)
MERGED = os.environ.get("MERGED", str(_REPO / "experiments/results/rl/_em_merged"))
PROFILE = os.environ.get("PROFILE", "stubborn")
SIZES, CKPTS = ["4B", "8B"], ["base", "gs50", "gs100"]


def ckpt_path(size, ckpt):
    """The checkpoint directory of a (size, ckpt) — the resolve_ckpt case block of ckpt_paths.sh."""
    if ckpt != "base":
        return f"{MERGED}/sender_qwen3-{size}_{PROFILE}_j35B_fakepen0_fmt0p1_{ckpt}"
    if size == "4B":
        return f"{TC}/Qwen3-4B-Instruct-2507"
    if size == "8B":
        p = f"{TC}/Qwen3-8B"
        return p if os.path.isdir(p) else f"{MODELS_DIR}/Qwen3-8B"
    return f"{TC}/Qwen3-{size}"


def effective_template(path):
    """-> (template_text, where_it_came_from). Prefers the standalone jinja the merger emits."""
    jinja = os.path.join(path, "chat_template.jinja")
    if os.path.exists(jinja):
        return open(jinja).read(), "chat_template.jinja"
    tok = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(tok):
        t = json.load(open(tok)).get("chat_template")
        if isinstance(t, list):                      # some configs ship a list of named templates
            t = json.dumps(t, sort_keys=True)
        if t:
            return t, "tokenizer_config.json[chat_template]"
    return None, "MISSING"


def record(path):
    t, src = effective_template(path)
    return {"path": path, "source": src, "len": len(t or ""),
            "sha256_16": hashlib.sha256(t.encode()).hexdigest()[:16] if t else None}


def _identical(per_ckpt):
    """True when every recorded prober of this size resolved to the same non-missing template."""
    hashes = {v["sha256_16"] for v in per_ckpt.values()}
    return len(hashes) == 1 and None not in hashes


_VERDICT = {True: "IDENTICAL across probers",
            False: "*** DIFFERS — the probe-step axis is confounded ***",
            None: "recorded (one checkpoint: nothing to compare it against yet)"}


def _report(report, ok, out_path, key):
    for size, per in report.items():
        print(f"--- {size}")
        for ck, v in per["per_ckpt"].items():
            print(f"  {ck:>6}: {v['sha256_16']}  len={v['len']:>6}  from {v['source']}")
        print(f"  => {_VERDICT[per[key]]}")
    payload = {"identical_all_sizes": ok, "sizes": report}
    if out_path:
        json.dump(payload, open(out_path, "w"), indent=2)
        print(f"[tpl-check] -> {out_path} (identical_all_sizes={ok})")
    return 0 if ok else 1


def verify_dir(ans_dir, sizes):
    """Replay the per-job records under `ans_dir` and apply the identity gate to them."""
    report, ok = {}, True
    for size in sizes:
        per = {}
        for f in sorted(glob.glob(os.path.join(ans_dir, f"template_check_{size}_*.json"))):
            d = json.load(open(f))
            per.update(d.get("sizes", {}).get(size, {}).get("per_ckpt", {}))
        if not per:
            print(f"[tpl-check] no template records for {size} in {ans_dir}")
            ok = False
            continue
        same = _identical(per) and len(per) > 1
        ok &= same
        report[size] = {"per_ckpt": per, "identical_across_probers": same,
                        "n_records": len(per)}
    return _report(report, ok, None, "identical_across_probers")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", default=None, help="JSON report to write")
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--ckpts", nargs="+", default=CKPTS)
    ap.add_argument("--path", default=None,
                    help="hash THIS directory instead of the resolved one (one --sizes and one "
                         "--ckpts): the directory the job actually serves")
    ap.add_argument("--verify-dir", default=None,
                    help="read the per-job records written there and apply the identity gate")
    args = ap.parse_args()

    if args.verify_dir:
        return verify_dir(args.verify_dir, args.sizes)
    if args.path and (len(args.sizes) != 1 or len(args.ckpts) != 1):
        ap.error("--path needs exactly one --sizes and one --ckpts")

    report, ok = {}, True
    for size in args.sizes:
        per = {ck: record(args.path or ckpt_path(size, ck)) for ck in args.ckpts}
        # A single checkpoint only records (and fails if its template is unreadable).
        same = _identical(per) if len(per) > 1 else None
        ok &= (per[args.ckpts[0]]["sha256_16"] is not None) if same is None else same
        report[size] = {"per_ckpt": per, "identical_across_probers": same,
                        "n_records": len(per)}
    return _report(report, ok, args.out, "identical_across_probers")


if __name__ == "__main__":
    sys.exit(main())
