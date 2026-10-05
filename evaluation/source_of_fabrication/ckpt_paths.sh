# Checkpoint resolver for the six policies of this experiment, sourced by both SLURM stages
# (generate_items.slurm, probe_cell.slurm) so they always resolve the same directories.
# Merged run names are the ones scripts/merge_verl_ckpt.slurm writes.
#
#   source ckpt_paths.sh; resolve_ckpt 8B gs100   # -> CK_PATH, CK_NAME
# Base checkpoints live under MODELS_DIR (default <repo>/models), merged ones under MERGED.
_SOF_REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$_SOF_REPO/models}"
TC="${TC:-$MODELS_DIR}"
MERGED="${MERGED:-$_SOF_REPO/experiments/results/rl/_em_merged}"
RECEIVER_PATH="${RECEIVER_PATH:-$TC/Qwen3.5-35B-A3B}"
RECEIVER_NAME="${RECEIVER_NAME:-qwen3.5-35B}"
PROFILE="${PROFILE:-stubborn}"
resolve_ckpt() {  # $1 = SIZE, $2 = base|gs50|gs100  -> sets CK_PATH, CK_NAME
  case "$2" in
    base)  case "$1" in
             4B)  CK_PATH="$TC/Qwen3-4B-Instruct-2507" ;;
             # The fallback is the Qwen3-8B copy scripts/rl_train_sender.slurm trains from.
             8B)  CK_PATH="$TC/Qwen3-8B"
                  [ -d "$CK_PATH" ] || CK_PATH="$MODELS_DIR/Qwen3-8B" ;;
             14B) CK_PATH="$TC/Qwen3-14B" ;;
             *)   echo "[ckpt] ERROR: unknown SIZE '$1'"; return 1 ;;
           esac; CK_NAME="qwen3-${1}-base" ;;
    gs50|gs100) CK_PATH="$MERGED/sender_qwen3-${1}_${PROFILE}_j35B_fakepen0_fmt0p1_${2}"
                CK_NAME="qwen3-${1}-${2}" ;;
    *) echo "[ckpt] ERROR: unknown CKPT '$2' (base|gs50|gs100)"; return 1 ;;
  esac
}
