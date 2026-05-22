#!/usr/bin/env bash
# RunPod setup script for ChoirSMT.
# Designed to run on a fresh RunPod GPU pod (PyTorch image).
#
# Modes:
#   bash runpod_setup.sh generate   — build the dataset
#   bash runpod_setup.sh train      — train the model on the dataset
#   bash runpod_setup.sh install    — just install deps (used by other modes)
#   bash runpod_setup.sh autopilot  — install + generate + train, all-in-one

set -euo pipefail

MODE="${1:-help}"
WORKSPACE="${WORKSPACE:-/workspace}"
DATASET_DIR="$WORKSPACE/dataset"
CHECKPOINTS_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

install_deps() {
  echo "[setup] installing system deps..." | tee -a "$LOG_DIR/setup.log"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # MuseScore 3 from official Ubuntu repo (headless-friendly, no AppImage)
  apt-get install -y -qq \
    git wget xvfb \
    musescore3 \
    fonts-freefont-ttf libfontconfig1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 \
    libxcb-xinerama0 libxkbcommon-x11-0 libxkbcommon0 \
    || echo "[setup] some apt packages failed but continuing"

  echo "[setup] installing Python deps..." | tee -a "$LOG_DIR/setup.log"
  pip install --quiet \
    music21 \
    Pillow \
    numpy \
    torch torchvision \
    transformers \
    huggingface_hub
  echo "[setup] done"
}

ensure_repo() {
  if [ ! -d "$WORKSPACE/code" ]; then
    echo "[repo] cloning code..." | tee -a "$LOG_DIR/setup.log"
    git clone https://github.com/rubenreut/choir-omr-pipeline.git "$WORKSPACE/code"
  fi
}

generate() {
  ensure_repo
  cd "$WORKSPACE/code"
  COUNT="${COUNT:-2000}"
  AUGMENTS="${AUGMENTS:-5}"
  echo "[generate] building dataset: $COUNT pieces × $AUGMENTS augments..." | tee -a "$LOG_DIR/generate.log"
  # MuseScore3 needs a display server even in CLI mode
  Xvfb :99 -screen 0 1024x768x24 &
  XVFB_PID=$!
  export DISPLAY=:99
  python3 build_dataset.py \
    --count "$COUNT" \
    --augments "$AUGMENTS" \
    --out "$DATASET_DIR" \
    --musescore /usr/bin/mscore3 \
    2>&1 | tee -a "$LOG_DIR/generate.log"
  kill $XVFB_PID 2>/dev/null || true
  echo "[generate] done. Dataset:" | tee -a "$LOG_DIR/generate.log"
  du -sh "$DATASET_DIR"/* | tee -a "$LOG_DIR/generate.log"
}

train() {
  ensure_repo
  cd "$WORKSPACE/code"
  BATCH="${BATCH:-8}"
  EPOCHS="${EPOCHS:-15}"
  echo "[train] starting training (batch=$BATCH, epochs=$EPOCHS)..." | tee -a "$LOG_DIR/train.log"
  python3 train.py \
    --manifest "$DATASET_DIR/manifest.json" \
    --vocab "$DATASET_DIR/vocab.json" \
    --image-root "$DATASET_DIR" \
    --output "$CHECKPOINTS_DIR" \
    --batch "$BATCH" \
    --epochs "$EPOCHS" \
    2>&1 | tee -a "$LOG_DIR/train.log"
  echo "[train] done. Checkpoints:" | tee -a "$LOG_DIR/train.log"
  ls -lh "$CHECKPOINTS_DIR" | tee -a "$LOG_DIR/train.log"
}

autopilot() {
  install_deps 2>&1 | tee -a "$LOG_DIR/autopilot.log"
  generate 2>&1 | tee -a "$LOG_DIR/autopilot.log"
  train 2>&1 | tee -a "$LOG_DIR/autopilot.log"
  echo "[autopilot] ALL DONE. Final model in $CHECKPOINTS_DIR" | tee -a "$LOG_DIR/autopilot.log"
}

case "$MODE" in
  install)   install_deps ;;
  generate)  install_deps; generate ;;
  train)     install_deps; train ;;
  autopilot) autopilot ;;
  *)
    echo "Usage: $0 {install|generate|train|autopilot}"
    echo "  Env vars: COUNT (default 2000), AUGMENTS (5), BATCH (8), EPOCHS (15)"
    exit 1
    ;;
esac
