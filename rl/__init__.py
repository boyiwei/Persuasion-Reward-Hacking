"""Sender-RL package: GRPO-trains the Sender so its 3-round arguments raise a simulated
Receiver's reported P(guilty) (paper §3.1 + App. C.2), with reward-hacking monitors.

Imported by offline data prep and inside the verl trainer (same `.venv`), so it puts the repo root
on sys.path and reuses `agents.*` / `evaluation.*` read-only.
"""
import sys
from pathlib import Path

# Repo root on sys.path so `agents.*` and `experiments.*` (namespace packages)
# import regardless of the CWD verl launches us from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
