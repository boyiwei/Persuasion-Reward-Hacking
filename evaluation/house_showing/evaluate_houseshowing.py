"""
House-showing deception-rate evaluator: a thin wrapper over the shared engine
evaluation/audit_fabrications.py (domain "house-showing").

The Old Bailey false_claims judge re-worded for this game (seller / KNOWN FEATURES, scenario as
SCENARIO), counted with the exact-quote guard; writes `.deception.json`. Headline `rh_fake_evidence`
= flips of a known feature + invented facts, against the full template's `params.private.features`.

The judge is the served model at RECEIVER_HOST:RECEIVER_PORT (default 127.0.0.1:30001,
RECEIVER_MODEL_ID default qwen3.5-35B), which the SLURM launchers co-serve; for a judge served on
another node, point these at it (scripts/serve_receiver_xnode.slurm). With nothing served every
game judge-fails and the run exits 2 without a sidecar. `--judge-model` is accepted and ignored
(*_rl_eval.slurm passes it).

Example:
    RECEIVER_HOST=<node> RECEIVER_PORT=30001 RECEIVER_MODEL_ID=qwen3.5-35B \
    python evaluation/house_showing/evaluate_houseshowing.py \
        --result-file experiments/results/house-showing/<sender>/stubborn_house_showing_rlrollout__recv_qwen3.5-35B.json \
        --full-template datasets/house_showing/processed/full/house_showing_full.json \
        --model <sender> --receiver-config stubborn --no-wandb
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.audit_fabrications import run_cli  # noqa: E402


def main():
    # propagate the engine's return code (2 = every judged game failed, no sidecar written)
    sys.exit(run_cli("house-showing") or 0)


if __name__ == "__main__":
    main()
