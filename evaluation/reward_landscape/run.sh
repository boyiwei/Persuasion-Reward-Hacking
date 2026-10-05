#!/bin/bash
# Gate the audit artifacts, then recompute the figure's stats json from them. CPU only; a few
# minutes, mostly parsing rollout transcripts.
#
# check_inputs.py runs first and its failure is fatal: the claim is that every cell came from one
# judge instance under the current judge code, and the analyzer will still produce plausible
# numbers from a partial or mixed directory.
#
#   bash evaluation/reward_landscape/run.sh
#   RESULTS_ROOT=<dir> bash evaluation/reward_landscape/run.sh
#   AUDIT_DIR=<...>/j<jobid>/by_result bash evaluation/reward_landscape/run.sh   # before `final`
#
# Env: RESULTS_ROOT (default $REPO/experiments/results) AUDIT_DIR (default
#      <results root>/old-bailey/_reward_landscape_audit/final/by_result) MANIFEST (default
#      <results root>/reward_landscape/manifest.tsv; the same variable the audit job reads, so a
#      subset manifest gates and analyses the cells it audited) PYTHON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"

echo "=== check_inputs ==="
"$PY" "$HERE/check_inputs.py"

echo "=== analyze_belief_correlation -> belief_correlation_stats.json ==="
"$PY" "$HERE/analyze_belief_correlation.py"
echo "=== done ==="
