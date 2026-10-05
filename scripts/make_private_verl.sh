#!/bin/bash
# Create or refresh a private verl checkout patched from this worktree, so a study never shares a
# mutable verl with another worktree.
#
# rl/patch_verl.sh edits one checkout in place, and the repo venv finds verl through a plain .pth
# path entry (__editable__.verl-0.7.0.dev0.pth -> <verl_dir>), not a MetaPathFinder, so PYTHONPATH
# wins over it. scripts/rl_train_sender.slurm locates the files it gates on by importing verl, so
# the launcher's VERL_HOME knob (which sets PYTHONPATH, inherited by Ray workers) redirects the
# patch greps, the patch checker and training together.
#
# Worktrees with byte-identical rl/patch_verl.sh can share the canonical checkout (re-patching is a
# no-op). Use a private one when they differ, or a patch change will rewrite verl under another
# worktree's queued jobs.
#
#   bash scripts/make_private_verl.sh <private_verl_dir> [source_verl_dir]   # source defaults to $VERL_DIR
#   sbatch --export=ALL,VERL_HOME=<private_verl_dir>,ALGO=base+aux_loss,PROMPT=strategies,DIST=stubborn,SFT_SPLIT=1 scripts/rl_train_sender.slurm
#
# About 770 MB per copy. Idempotent: an existing destination is re-patched, not re-copied.
set -euo pipefail

DEST="${1:?usage: make_private_verl.sh <dest-verl-dir> [source-verl-dir]}"
SRC="${2:-${VERL_DIR:?set VERL_DIR to a verl 0.7.0.dev checkout, or pass the source as the second argument}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -d "$SRC/verl" ]] || { echo "[private-verl] ERROR: source is not a verl checkout: $SRC"; exit 1; }
[[ -f "$REPO/rl/patch_verl.sh" ]] || { echo "[private-verl] ERROR: run from a repo worktree"; exit 1; }
[[ "$DEST" != "$SRC" ]] || { echo "[private-verl] ERROR: dest must differ from source"; exit 1; }

if [[ -d "$DEST/verl" ]]; then
  echo "[private-verl] reusing existing checkout: $DEST"
else
  avail_kb=$(df -Pk "$(dirname "$DEST")" | awk 'NR==2{print $4}')
  (( avail_kb > 2000000 )) || { echo "[private-verl] ERROR: <2GB free at $(dirname "$DEST")"; exit 1; }
  echo "[private-verl] copying $SRC -> $DEST (about 770 MB)"
  mkdir -p "$(dirname "$DEST")"
  cp -a "$SRC" "$DEST.partial.$$"
  mv "$DEST.partial.$$" "$DEST"
fi

echo "[private-verl] patching with $REPO/rl/patch_verl.sh"
bash "$REPO/rl/patch_verl.sh" "$DEST"

PY="${PY:-$REPO/.venv/bin/python}"
got="$(PYTHONPATH="$DEST" "$PY" -c 'import verl, os; print(os.path.dirname(os.path.dirname(verl.__file__)))')"
[[ "$got" == "$DEST" ]] || {
  echo "[private-verl] ERROR: PYTHONPATH did not win; import verl resolved to $got"; exit 1; }

# Same gate as a norm_ratio run's preflight.
PYTHONPATH="$DEST" "$PY" -m rl.check_aux_grad_balance_patch \
  --actor "$DEST/verl/workers/actor/dp_actor.py" \
  --trainer "$DEST/verl/trainer/ppo/ray_trainer.py" \
  --launcher "$REPO/scripts/rl_train_sender.slurm"

echo "[private-verl] OK: $DEST"
echo "[private-verl] use it with:  VERL_HOME=$DEST"
