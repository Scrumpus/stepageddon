#!/usr/bin/env bash
# RunPod bootstrap for stepageddon chart-generator training.
#
# One-shot setup: clones the repo (if needed), syncs deps, unzips training
# data, and kicks off training inside tmux so SSH disconnects don't kill it.
#
# Usage (on a fresh RunPod pod, as root):
#   curl -fsSL <raw-url-of-this-script> | bash -s -- \
#       --repo https://github.com/<you>/stepageddon.git \
#       --branch improve-model
#
# Or, after scp'ing the repo manually:
#   bash backend/ml/runpod_setup.sh --skip-clone
#
# Optional flags:
#   --repo <url>           Git URL to clone (required unless --skip-clone)
#   --branch <name>        Branch to check out (default: main)
#   --workdir <path>       Where to put the repo (default: /workspace/stepageddon)
#   --data-zip <path>      Override path to training_data.zip
#   --skip-clone           Skip git clone (assume repo is already at --workdir)
#   --skip-train           Set up everything but don't launch training
#   --wandb-key <key>      Set WANDB_API_KEY for the session
#   --epochs <n>           Override --epochs passed to train.py
#   --batch-size <n>       Override --batch-size passed to train.py
#   --extra-args "..."     Extra args appended verbatim to train.py
#
# After this finishes, attach to the running session with:
#   tmux attach -t train
# Detach with Ctrl+B then D.

set -euo pipefail

REPO_URL=""
BRANCH="main"
WORKDIR="/workspace/stepageddon"
DATA_ZIP=""
SKIP_CLONE=0
SKIP_TRAIN=0
WANDB_KEY=""
EPOCHS="80"
BATCH_SIZE="32"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --data-zip) DATA_ZIP="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --wandb-key) WANDB_KEY="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m[runpod-setup]\033[0m %s\n' "$*"; }

log "GPU sanity check"
nvidia-smi || { echo "No GPU visible — wrong pod template?" >&2; exit 1; }

log "Installing system deps"
apt-get update -qq
apt-get install -y -qq tmux git unzip ffmpeg curl >/dev/null

if [[ "$SKIP_CLONE" -eq 0 ]]; then
  if [[ -z "$REPO_URL" ]]; then
    echo "--repo is required unless --skip-clone is set" >&2
    exit 1
  fi
  log "Cloning $REPO_URL @ $BRANCH into $WORKDIR"
  mkdir -p "$(dirname "$WORKDIR")"
  if [[ -d "$WORKDIR/.git" ]]; then
    git -C "$WORKDIR" fetch --depth 1 origin "$BRANCH"
    git -C "$WORKDIR" checkout "$BRANCH"
    git -C "$WORKDIR" reset --hard "origin/$BRANCH"
  else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
  fi
fi

cd "$WORKDIR/backend"

log "Installing Python deps via uv"
pip install --quiet uv
uv sync

log "Activating virtualenv"
# uv places the venv at backend/.venv
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing wandb (optional, no-op if not used)"
pip install --quiet wandb

if [[ -n "$WANDB_KEY" ]]; then
  export WANDB_API_KEY="$WANDB_KEY"
  log "WANDB_API_KEY set for this session"
fi

DATA_DIR="$WORKDIR/backend/ml/training_data"
if [[ ! -d "$DATA_DIR" || -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
  ZIP_PATH="${DATA_ZIP:-$WORKDIR/backend/ml/training_data.zip}"
  if [[ -f "$ZIP_PATH" ]]; then
    log "Unzipping $ZIP_PATH"
    mkdir -p "$DATA_DIR"
    unzip -q -o "$ZIP_PATH" -d "$WORKDIR/backend/ml/"
  else
    echo "Training data missing. Either upload training_data.zip to $WORKDIR/backend/ml/ or pass --data-zip." >&2
    exit 1
  fi
else
  log "Training data already present at $DATA_DIR"
fi

CKPT_DIR="$WORKDIR/backend/ml/checkpoints"
mkdir -p "$CKPT_DIR"

if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  log "Setup complete. --skip-train set; not launching training."
  log "To start training manually:"
  echo "  tmux new -s train"
  echo "  cd $WORKDIR/backend && source .venv/bin/activate"
  echo "  python -m ml.train --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE $EXTRA_ARGS"
  exit 0
fi

log "Launching training in tmux session 'train'"
TRAIN_CMD="cd $WORKDIR/backend && source .venv/bin/activate && python -m ml.train --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE $EXTRA_ARGS 2>&1 | tee $CKPT_DIR/train.log"

tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "$TRAIN_CMD"

log "Training started. Attach with:  tmux attach -t train"
log "Live log:                       tail -f $CKPT_DIR/train.log"
log "Checkpoints will land in:       $CKPT_DIR"
