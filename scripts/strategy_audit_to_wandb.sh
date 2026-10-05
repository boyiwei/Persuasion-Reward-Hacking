#!/bin/bash
# Push the offline persuasion-strategy audit (rl.strategy_audit.run output) into the original RL wandb
# runs. Run it on a machine with network access to wandb.
#
# Resumes each run by its discovered id and logs persuasion_strategy/<slug> per train step (x-axis
# persuasion_strategy/step). Validate with --dry-run first (logs to a throwaway run).
#
#   bash scripts/strategy_audit_to_wandb.sh --dry-run sender_qwen3-4B_stubborn_j35B_fakepen0   # validate
#   bash scripts/strategy_audit_to_wandb.sh sender_qwen3-4B_stubborn_j35B_fakepen0 [<run> ...]  # backfill
#
# Env: PYTHON (default <repo>/.venv/bin/python), RESULTS_ROOT (default <repo>/experiments/results/rl), REPO_ROOT (dir holding wandb/ for run-id
#      discovery; default <repo>), WANDB_ENTITY, WANDB_PROJECT (overrides auto-discovered project).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "ERROR: python not found: $PY (run 'uv sync --frozen' or set PYTHON)"; exit 1; }

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; shift; fi
[ "$#" -ge 1 ] || { echo "Usage: $0 [--dry-run] <run> [<run> ...]"; exit 1; }

PROJ_ARG=(); [ -n "${WANDB_PROJECT:-}" ] && PROJ_ARG=(--project "$WANDB_PROJECT")

for RUN in "$@"; do
  echo "=== strategy -> wandb: $RUN ${DRY} ==="
  "$PY" -m rl.strategy_audit.to_wandb --run "$RUN" $DRY "${PROJ_ARG[@]}"
done
