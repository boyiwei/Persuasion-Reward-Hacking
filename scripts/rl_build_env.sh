#!/bin/bash
# Install verl into the repo .venv (no second venv) and check that the repo's transformers-5.6 /
# sglang-0.5.12 / torch-2.11 stack (which knows `qwen3_5`) loads the Qwen3.5 checkpoints.
# The one-GRPO-step gate is scripts/rl_train_sender.slurm with SMOKE=1 (needs the 35B judge).
#
# Run on a machine with network access (uv pip downloads packages; the checks are CPU-only):
#   VERL_DIR=/path/to/verl bash scripts/rl_build_env.sh
#
# Env: VERL_DIR (required: a verl 0.7.0.dev checkout), POLICY_PATH (Qwen3.5-9B checkpoint, default
# MODELS_DIR/Qwen3.5-9B with MODELS_DIR defaulting to <repo>/models), CACHE_DIR (<repo>/.cache: uv,
# HF and XDG caches), RL_VENV (repo .venv).
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p logs

RL_VENV="${RL_VENV:-$REPO/.venv}"
PY="$RL_VENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
VERL_DIR="${VERL_DIR:?set VERL_DIR to a verl 0.7.0.dev checkout}"
POLICY_PATH="${POLICY_PATH:-$MODELS_DIR/Qwen3.5-9B}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_DIR/uv}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # local checkpoints only
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"

[[ -x "$PY" ]] || { echo "[env] ERROR: repo .venv missing at $RL_VENV (run 'uv sync --frozen')."; exit 1; }
[[ -d "$VERL_DIR" ]] || { echo "[env] ERROR: verl checkout not found at $VERL_DIR (set VERL_DIR)."; exit 1; }
# rl/patch_verl.sh targets verl 0.7.0.dev (commit 6dc50993).
[[ "$(cat "$VERL_DIR/verl/version/version" 2>/dev/null)" == 0.7.0.dev ]] \
  || { echo "[env] ERROR: $VERL_DIR is not verl 0.7.0.dev, which rl/patch_verl.sh targets."; exit 1; }

echo "[env] patching verl source for compat (numpy cap, AutoModelForVision2Seq, FSDP set->list) ..."
bash "$REPO/rl/patch_verl.sh" "$VERL_DIR"

# verl's dependencies are pinned in pyproject.toml / uv.lock and installed by `uv sync --frozen`, so
# verl itself goes in with --no-deps and cannot move any locked package.
echo "[env] installing verl (editable, --no-deps) into the repo .venv ..."
uv pip install --python "$PY" --no-deps -e "$VERL_DIR"
bash "$REPO/rl/patch_sglang.sh" "$PY"   # torch-2.11 weight-sync bounds-check

echo "[env] checking the venv against uv.lock ..."
UV_PROJECT_ENVIRONMENT="$RL_VENV" uv sync --frozen --inexact --check \
  || { echo "[env] ERROR: $RL_VENV differs from uv.lock (run 'uv sync --frozen --inexact')."; exit 1; }
"$PY" -c "import transformers, sglang, torch, verl; print('transformers', transformers.__version__, '| sglang', sglang.__version__, '| torch', torch.__version__, '| verl', verl.__version__)"

echo "[env] verifying transformers recognizes the Qwen3.5-9B checkpoint (qwen3_5) ..."
POLICY_PATH="$POLICY_PATH" "$PY" - <<'PYEOF'
import os, torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM
path = os.environ["POLICY_PATH"]
cfg = AutoConfig.from_pretrained(path, trust_remote_code=False)
print("architectures:", getattr(cfg, "architectures", None), "| model_type:", cfg.model_type)
# Qwen3_5Config is a composite VLM config (vocab_size lives under text_config), so
# AutoModelForCausalLM.from_config(cfg) crashes. Resolve the on-disk VLM class the way verl does,
# then the text tower from the text sub-config.
with torch.device("meta"):
    full = AutoModelForImageTextToText.from_config(cfg)
    AutoModelForCausalLM.from_config(cfg.get_text_config())   # model_type qwen3_5_text
print("[env] OK:", type(full).__name__, "(+ text tower) resolved under transformers",
      __import__("transformers").__version__)
PYEOF

echo "[env] re-running the repo's own import smoke (inference stack intact) ..."
"$PY" -c "import hydra, omegaconf, openai, wandb; import agents.rollout, agents.agent_quality; print('[env] repo imports OK')"

echo "[env] DONE. Next gate: sbatch --export=ALL,SMOKE=1,DIST=bayesian scripts/rl_train_sender.slurm"
