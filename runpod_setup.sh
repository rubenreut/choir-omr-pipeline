#!/usr/bin/env bash
# RunPod setup script for ChoirSMT training.
# Designed to run on a fresh Ubuntu RunPod instance with Python 3.10+ and a GPU.
#
# Usage on RunPod:
#   1. Spin up a CPU instance (cheap) for data generation: ~$0.10/hr
#      Run: bash runpod_setup.sh generate
#      That builds the dataset in /workspace/dataset/
#
#   2. Spin up an A100 GPU instance for training: ~$1.50-2/hr
#      Run: bash runpod_setup.sh train
#      Reads the dataset from the persistent volume, trains the model
#
#   3. Download trained model when done:
#      scp ... checkpoints/ckpt_epoch_XXX.pt ~/Desktop/choir/

set -euo pipefail

MODE="${1:-help}"
WORKSPACE="${WORKSPACE:-/workspace}"
DATASET_DIR="$WORKSPACE/dataset"
CHECKPOINTS_DIR="$WORKSPACE/checkpoints"

install_deps() {
  echo "[setup] installing system deps..."
  apt-get update -qq
  apt-get install -y -qq \
    git wget unzip xvfb libnss3 libgconf-2-4 libxi6 \
    libxss1 libxtst6 libasound2 libsecret-1-0 libgbm1

  echo "[setup] installing MuseScore (AppImage)..."
  if [ ! -f /usr/local/bin/mscore ]; then
    wget -q https://cdn.jsdelivr.net/musescore/MuseScore-Studio-4.4.4.243461245-x86_64.AppImage -O /tmp/mscore.AppImage
    chmod +x /tmp/mscore.AppImage
    mv /tmp/mscore.AppImage /usr/local/bin/mscore
  fi

  echo "[setup] installing Python deps..."
  pip install --quiet \
    music21==9.* \
    Pillow \
    numpy \
    torch torchvision \
    transformers \
    huggingface_hub

  echo "[setup] done."
}

generate() {
  install_deps
  COUNT="${COUNT:-30000}"
  AUGMENTS="${AUGMENTS:-5}"
  echo "[generate] building dataset: $COUNT pieces × $AUGMENTS augments..."
  cd "$(dirname "$0")"
  # Headless MuseScore needs an X server
  Xvfb :99 -screen 0 1024x768x24 &
  export DISPLAY=:99
  python3 build_dataset.py \
    --count "$COUNT" \
    --augments "$AUGMENTS" \
    --out "$DATASET_DIR" \
    --musescore /usr/local/bin/mscore
  echo "[generate] dataset built. Total disk:"
  du -sh "$DATASET_DIR"
}

train() {
  install_deps
  BATCH="${BATCH:-8}"
  EPOCHS="${EPOCHS:-50}"
  echo "[train] starting training..."
  cd "$(dirname "$0")"
  python3 train.py \
    --manifest "$DATASET_DIR/manifest.json" \
    --vocab "$DATASET_DIR/vocab.json" \
    --image-root "$DATASET_DIR" \
    --output "$CHECKPOINTS_DIR" \
    --batch "$BATCH" \
    --epochs "$EPOCHS"
  echo "[train] done. checkpoints in $CHECKPOINTS_DIR"
}

case "$MODE" in
  generate) generate ;;
  train) train ;;
  install) install_deps ;;
  *)
    echo "Usage: $0 {generate|train|install}"
    echo "  generate  — build the dataset (run on cheap CPU instance)"
    echo "  train     — train the model (run on GPU instance)"
    echo "  install   — just install deps"
    exit 1
    ;;
esac
