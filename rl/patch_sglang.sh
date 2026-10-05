#!/bin/bash
# Idempotent patch to the installed sglang (in the repo .venv) for verl hybrid-engine weight sync
# under torch 2.11. Called by scripts/rl_build_env.sh after the verl install. Usage:
#   bash rl/patch_sglang.sh <python>     (default: ./.venv/bin/python)
set -euo pipefail
PY="${1:-./.venv/bin/python}"
"$PY" - <<'PYEOF'
import sglang, os
p = os.path.join(os.path.dirname(sglang.__file__), "srt", "utils", "patch_torch.py")
s = open(p).read()
old = ('def _modify_tuple(t, index: int, modifier: Callable):\n'
       '    return *t[:index], modifier(t[index]), *t[index + 1 :]')
new = ('def _modify_tuple(t, index: int, modifier: Callable):\n'
       '    # torch>=2.11 reduce_tensor returns a SHORT tuple for non-CUDA tensors (no device at idx 6);\n'
       '    # only CUDA tensors carry the device index. Skip modification when out of range.\n'
       '    if index >= len(t):\n        return t\n'
       '    return *t[:index], modifier(t[index]), *t[index + 1 :]')
if "Skip modification when out of range" in s:
    print("[patch_sglang] _modify_tuple bounds-check: already present")
elif old in s:
    open(p, "w").write(s.replace(old, new)); print("[patch_sglang] _modify_tuple bounds-check: applied")
else:
    raise SystemExit("[patch_sglang] anchor not found (sglang version drift?)")
PYEOF
echo "[patch_sglang] done"
