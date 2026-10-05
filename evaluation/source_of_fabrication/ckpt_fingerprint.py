#!/usr/bin/env python3
"""Checkpoint fingerprint: sha256 of every regular file in a checkpoint directory.

Enforces that generation and evaluation use the same checkpoints: stage A writes the fingerprint
of the sender it served (rollouts/_ckpt__*.json), and probe_cell.slurm recomputes it for the
prober and refuses to run on any difference (--expect).

  python ckpt_fingerprint.py --path <ckpt dir> --out fp.json [--expect other_fp.json]

Exit 1 on mismatch. Hashing a multi-GB checkpoint takes a while, so the SLURM jobs hash while the
server loads.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(root):
    root = os.path.realpath(root)
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, root)
            files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    digest = hashlib.sha256("\n".join(f"{k} {v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"path": root, "basename": os.path.basename(root.rstrip("/")), "n_files": len(files),
            "total_bytes": sum(v["bytes"] for v in files.values()), "fingerprint": digest,
            "files": files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--expect", default=None,
                    help="another fingerprint JSON; every file must match (sha256 + size)")
    args = ap.parse_args()

    t0 = time.time()
    fp = fingerprint(args.path)
    fp["label"] = args.label
    fp["hashed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fp["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(fp, open(args.out, "w"), indent=2)
    print(f"[fp] {args.label or fp['basename']}: {fp['n_files']} files, {fp['total_bytes'] / 2**30:.2f} GiB, "
          f"fingerprint {fp['fingerprint'][:16]}… ({fp['seconds']} s) -> {args.out}")

    rc = 0
    if args.expect:
        ref = json.load(open(args.expect))
        a, b = fp["files"], ref.get("files", {})
        diff = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b)
                                                if a[k]["sha256"] != b[k]["sha256"] or a[k]["bytes"] != b[k]["bytes"])
        if diff:
            print(f"[fp] ERROR: {len(diff)} file(s) differ from {args.expect}: {diff[:6]}")
            rc = 1
        else:
            print(f"[fp] MATCH: identical to {args.expect} ({ref.get('label') or ref.get('basename')}, "
                  f"fingerprint {ref.get('fingerprint', '')[:16]}…)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
