#!/bin/bash
#SBATCH --job-name=sft_train_sender
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Sender SFT (stage 4 of the SFT-then-RL curriculum): full-FT the policy on the audited-clean
# gpt-5.4-mini demonstrations (datasets/old_bailey/sft/build_sft_dataset.py) with verl's multiturn
# fsdp_sft_trainer. Loss is on each prefix example's final assistant turn (loss_mask override);
# sft/sft_dataset.py PersuasionMultiTurnSFTDataset fixes the Qwen3 chat template for prefix
# consistency and GRPO parity. Pure FSDP training on N_GPUS, no serving.
#
# The final checkpoint includes a full HF model at <SAVE_DIR>/global_step_N/huggingface/, symlinked
# as <SAVE_DIR>/hf_final. Feed it to GRPO:
#   sbatch --export=ALL,DIST=stubborn,PROMPT=strategies,SFT_SPLIT=1,SFT_INIT=<SAVE_DIR>/hf_final \
#          scripts/rl_train_sender.slurm
# If the HF save fails: python -m verl.model_merger merge --backend fsdp
#   --local_dir <SAVE_DIR>/global_step_N --target_dir <out>   (no actor/ subdir, unlike GRPO ckpts)
#
# Submit:
#   sbatch --export=ALL,POLICY_MODEL=qwen3-4B scripts/sft_train_sender.slurm
#   sbatch --export=ALL,SMOKE=1,POLICY_MODEL=qwen3-4B scripts/sft_train_sender.slurm   # 2-step smoke (2 GPUs)
#
# Env, as NAME (default): meaning.
#   POLICY_MODEL (qwen3-4B): qwen3-4B | qwen3-8B | qwen3-14B; POLICY_PATH overrides the checkpoint.
#   MODELS_DIR (<repo>/models): default checkpoints are MODELS_DIR/<Model> (Qwen3-4B-Instruct-2507,
#     Qwen3-8B, Qwen3-14B). CACHE_DIR (<repo>/.cache): HF, wandb, XDG, Triton and inductor caches.
#   DATA_DIR (<repo>/datasets/old_bailey/_generated/sft/strategies_stubborn): SFT parquets.
#   DATASET_PY, DATASET_CLS (sft/sft_dataset.py, PersuasionMultiTurnSFTDataset): data.custom_cls.
#   EPOCHS (2); LR (1e-5; sft_trainer.yaml schedule: cosine, 10% warmup, wd 0.01).
#   TRAIN_BS (32; verl normalizes per rank), MICRO_BS (1 per GPU), N_GPUS (4), TEST_FREQ (10).
#   MAX_LEN (8192): truncation=error is a deliberate tripwire; the builder's --max-tokens keeps rows under it.
#   EXP_NAME, RL_VENV, WANDB_MODE (offline), EXTRA_OVERRIDES.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; mkdir -p logs
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"

RL_VENV="${RL_VENV:-$REPO/.venv}"; PY="$RL_VENV/bin/python"
SMOKE="${SMOKE:-0}"
# Caller-set values, honoured by the SMOKE block below.
N_GPUS_EXPLICIT="${N_GPUS:-}"; TRAIN_BS_EXPLICIT="${TRAIN_BS:-}"
POLICY_MODEL="${POLICY_MODEL:-qwen3-4B}"
case "$POLICY_MODEL" in
  qwen3-4B)  _POLICY_DEFAULT="$MODELS_DIR/Qwen3-4B-Instruct-2507" ;;
  qwen3-8B)  _POLICY_DEFAULT="$MODELS_DIR/Qwen3-8B" ;;
  qwen3-14B) _POLICY_DEFAULT="$MODELS_DIR/Qwen3-14B" ;;
  *) echo "POLICY_MODEL must be qwen3-4B|qwen3-8B|qwen3-14B (or pass POLICY_PATH explicitly)"; exit 1 ;;
esac
POLICY_PATH="${POLICY_PATH:-$_POLICY_DEFAULT}"
DATA_DIR="${DATA_DIR:-$REPO/datasets/old_bailey/_generated/sft/strategies_stubborn}"
# data.custom_cls; the default is the demonstration-SFT class.
DATASET_PY="${DATASET_PY:-$REPO/sft/sft_dataset.py}"
DATASET_CLS="${DATASET_CLS:-PersuasionMultiTurnSFTDataset}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
TRAIN_BS="${TRAIN_BS:-32}"
MICRO_BS="${MICRO_BS:-1}"      # 1/GPU: the SFT loss materializes full [b*s, vocab] logits (no chunking)
MAX_LEN="${MAX_LEN:-8192}"
N_GPUS="${N_GPUS:-4}"
TEST_FREQ="${TEST_FREQ:-10}"   # -1 or >0, never 0 (global_step % test_freq)
_VAL_FILES_EXPLICIT="${VAL_FILES:-}"        # empty = default path
VAL_FILES="${VAL_FILES:-$DATA_DIR/sft_val.parquet}"
# A --val-games 0 dataset has no holdout, but verl still builds a val loader: point it at the train
# file and disable validation (with save_freq=-1 only the last step is saved, so val loss selects
# nothing). Default path only: a missing explicit VAL_FILES is a typo and must hit the guard below.
if [[ -z "$_VAL_FILES_EXPLICIT" && ! -f "$VAL_FILES" && -f "$DATA_DIR/sft_train.parquet" ]]; then
  echo "[sft] no sft_val.parquet in $DATA_DIR -- no-holdout dataset; validating off the train file with TEST_FREQ=-1"
  VAL_FILES="$DATA_DIR/sft_train.parquet"; TEST_FREQ=-1
fi
SMOKE_SUFFIX=""; SMOKE_OVERRIDES=""
if [[ "$SMOKE" == 1 ]]; then
  SMOKE_SUFFIX="_smoke"
  # Honour an explicit N_GPUS/TRAIN_BS so a smoke can test a real batch shape (a balanced-sampler smoke
  # needs a batch divisible by its 4 strata, e.g. TRAIN_BS=8); otherwise 2 GPUs, bs 2.
  N_GPUS="${N_GPUS_EXPLICIT:-2}"; TRAIN_BS="${TRAIN_BS_EXPLICIT:-2}"
  # tiny smoke data can't feed a dp=4 drop_last val loader -> validate on the train file
  VAL_FILES="$DATA_DIR/sft_train.parquet"
  SMOKE_OVERRIDES="trainer.total_training_steps=2"
fi
EXP_NAME="${EXP_NAME:-sft_${POLICY_MODEL}_$(basename "$DATA_DIR")_ep${EPOCHS}${SMOKE_SUFFIX}}"
SAVE_DIR="$REPO/experiments/results/sft/$EXP_NAME"

module purge || true
export NO_PROXY="localhost,127.0.0.1,::1,${HOSTNAME:-localhost}"; export no_proxy="$NO_PROXY"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_DIR/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_DIR/inductor}"
# wandb runs offline by default; upload the run afterwards with `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-persuasion-gym-oldbailey-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

[[ -x "$PY" ]] || { echo "ERROR: repo .venv missing at $RL_VENV"; exit 1; }
# Check GPUs before `import verl`, which initializes CUDA: on a node where torch sees fewer devices
# than allocated, the import dies and the patch check below would misreport a missing verl patch.
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
(( NGPU >= N_GPUS )) || { echo "ERROR: need $N_GPUS GPUs, have $NGPU. Resubmit with --gres=gpu:$N_GPUS."; exit 1; }
"$PY" -c "import torch; n=torch.cuda.device_count(); assert n >= $N_GPUS, n" 2>/dev/null \
  || { echo "ERROR: torch sees fewer than $N_GPUS CUDA devices on $(hostname) though nvidia-smi lists $NGPU."
       echo "       This node is misconfigured -- resubmit; it will land elsewhere."; exit 1; }
"$PY" -c "import verl" 2>/dev/null || { echo "ERROR: verl not installed in .venv -- run scripts/rl_build_env.sh first."; exit 1; }
# patches 11 (sft attn_implementation knob) + 12 (multiturn BatchEncoding unwrap) must both be applied
grep -q "persuasion-gym\] (11)" "$("$PY" -c 'import verl.trainer.fsdp_sft_trainer as m; print(m.__file__)')" \
  || { echo "ERROR: verl patch (11) missing -- run: bash rl/patch_verl.sh"; exit 1; }
grep -q "persuasion-gym\] (12)" "$("$PY" -c 'import verl.utils.dataset.multiturn_sft_dataset as m; print(m.__file__)')" \
  || { echo "ERROR: verl patch (12) missing -- run: bash rl/patch_verl.sh"; exit 1; }
[[ -f "$DATA_DIR/sft_train.parquet" ]] || { echo "ERROR: no sft_train.parquet in $DATA_DIR -- run datasets/old_bailey/sft/build_sft_dataset.py first"; exit 1; }
[[ -f "$DATASET_PY" ]] || { echo "ERROR: DATASET_PY does not exist: $DATASET_PY"; exit 1; }
# Match a real class definition: a substring grep accepts DATASET_CLS=Foo for `class FooBar`, and
# the typo would only surface after the job has its GPUs.
grep -Eq "^[[:space:]]*class[[:space:]]+${DATASET_CLS}[[:space:]]*[(:]" "$DATASET_PY" \
  || { echo "ERROR: $DATASET_PY defines no class $DATASET_CLS"; exit 1; }
[[ -f "$VAL_FILES" ]] || { echo "ERROR: val parquet missing: $VAL_FILES"; exit 1; }
# Last guard, so a failed validation never leaves an empty claimed SAVE_DIR. Claim it atomically with
# mkdir (no -p): resume_mode=disable turns an existing SAVE_DIR into a silent overwrite of a checkpoint
# a GRPO arm may have been initialized from, and duplicate submissions would corrupt it
# (--dependency=singleton would serialize them, so the second job would retrain instead of standing
# down). Create the parent first: mkdir makes only the leaf, and a missing experiments/results/sft
# would read as a lost claim.
mkdir -p "$(dirname "$SAVE_DIR")"
if ! mkdir "$SAVE_DIR" 2>/dev/null; then
  if [[ "${FORCE_OVERWRITE:-0}" != 1 ]]; then
    # A duplicate submission that lost the claim exits 0; a stale dir exits 1, so sacct never shows
    # COMPLETED for a checkpoint that was never written.
    _owner=""
    [[ -f "$SAVE_DIR/.claim" ]] && _owner="$(tr -dc '0-9' < "$SAVE_DIR/.claim")"
    if [[ -n "$_owner" ]] && squeue -h -j "$_owner" -o "%t" 2>/dev/null | grep -q .; then
      echo "[sft] $SAVE_DIR is claimed by job $_owner, which is still running."
      echo "[sft] This job did not take the claim -- nothing to do."
      exit 0
    fi
    echo "ERROR: $SAVE_DIR already exists and no live job owns it${_owner:+ (stale claim by $_owner)}."
    echo "       resume_mode=disable would OVERWRITE it, destroying a checkpoint a GRPO arm may"
    echo "       have been initialized from. Pick a new EXP_NAME, remove it, or re-submit with"
    echo "       FORCE_OVERWRITE=1."
    exit 1
  fi
  echo "[sft] FORCE_OVERWRITE=1 -- reusing existing $SAVE_DIR"
fi
echo "${SLURM_JOB_ID:-0}" > "$SAVE_DIR/.claim"

echo "[sft] policy=$POLICY_MODEL ($POLICY_PATH) data=$DATA_DIR epochs=$EPOCHS lr=$LR bs=$TRAIN_BS"
echo "[sft] dataset_cls=$DATASET_CLS ($DATASET_PY) max_len=$MAX_LEN"

# Provenance next to the weights. Downstream launchers gate on dataset_cls: whole-turn and
# answer-token checkpoints differ, and the directory name cannot show it.
mkdir -p "$SAVE_DIR"
cat > "$SAVE_DIR/run_config.json" <<EOF
{
  "exp_name": "$EXP_NAME",
  "policy_model": "$POLICY_MODEL",
  "policy_path": "$POLICY_PATH",
  "data_dir": "$DATA_DIR",
  "dataset_py": "$DATASET_PY",
  "dataset_cls": "$DATASET_CLS",
  "epochs": "$EPOCHS",
  "lr": "$LR",
  "train_batch_size": "$TRAIN_BS",
  "micro_batch_size_per_gpu": "$MICRO_BS",
  "max_length": "$MAX_LEN",
  "n_gpus": "$N_GPUS",
  "smoke": "$SMOKE",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "submit_dir": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF
echo "[sft] exp=$EXP_NAME -> $SAVE_DIR"

"$RL_VENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/sft_train.parquet" \
  data.val_files="$VAL_FILES" \
  data.multiturn.enable=true \
  data.custom_cls.path="$DATASET_PY" \
  data.custom_cls.name="$DATASET_CLS" \
  data.max_length="$MAX_LEN" \
  data.truncation=error \
  data.train_batch_size="$TRAIN_BS" \
  data.micro_batch_size_per_gpu="$MICRO_BS" \
  model.partial_pretrain="$POLICY_PATH" \
  +model.attn_implementation=sdpa \
  optim.lr="$LR" \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq=-1 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.resume_mode=disable \
  'trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  ${SMOKE_OVERRIDES} \
  ${EXTRA_OVERRIDES:-}

# SFT_INIT handle: the highest-step HF checkpoint. Sort on the numeric step only; SAVE_DIR contains
# underscores, so `sort -t_ -kN` on the full path would mis-order.
_LAST_STEP="$(ls -d "$SAVE_DIR"/global_step_* 2>/dev/null | sed 's#.*/global_step_##' | sort -n | tail -1)"
[[ -n "$_LAST_STEP" ]] || { echo "ERROR: no global_step_* checkpoint under $SAVE_DIR"; exit 1; }
LAST="$SAVE_DIR/global_step_$_LAST_STEP"
ls "$LAST"/huggingface/*.safetensors >/dev/null 2>&1 \
  || { echo "ERROR: $LAST/huggingface has no safetensors (hf_model save failed?) -- use the model_merger fallback (header)."; exit 1; }
ln -sfn "$LAST/huggingface" "$SAVE_DIR/hf_final"
echo "[sft] done. SFT_INIT handle: $SAVE_DIR/hf_final"
