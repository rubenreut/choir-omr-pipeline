# ChoirSMT — DIY OMR for SATB choir scores

Custom end-to-end OMR pipeline. Generates synthetic SATB training data, trains
an ImageNet-backbone transformer model, deploys to the choir server.

## What's in here

| File | Purpose |
|------|---------|
| `synthesize.py` | Generate random but musically-plausible SATB MusicXML pieces |
| `extract_music21.py` | Pull Bach chorales from music21 corpus (optional, real music supplement) |
| `render.py` | Render MusicXML → PNG via MuseScore CLI |
| `augment.py` | Apply phone-photo-like distortions (rotation, lighting, blur) |
| `tokenize_satb.py` | MusicXML → token sequence (kern-style) and vocab builder |
| `dataset.py` | PyTorch Dataset for (image, tokens) pairs |
| `train.py` | Training script: ConvNeXt-Tiny + transformer decoder |
| `build_dataset.py` | End-to-end: generate → render → augment → tokenize → manifest |
| `runpod_setup.sh` | Cloud setup + run scripts |

## Architecture

- **Visual encoder**: ConvNeXt-Tiny (ImageNet pretrained, free from torchvision)
- **Decoder**: 6-layer transformer, d=512, 8 heads, trained from scratch on SATB tokens
- **Total params**: ~80M (~160 MB FP16, ~80 MB INT8)
- **Inference**: ~1–2 sec/page on A100, ~10–15 sec/page on CPU

## Workflow

### 1. Local development (Mac, free)

Test the pipeline on a tiny dataset:
```sh
# Generate 5 pieces
python3 synthesize.py --count 5 --seed 42
# Render
for f in musicxml_synth/*.musicxml; do
  mscore -o "rendered/$(basename $f .musicxml).png" "$f"
done
# Tokenize one
python3 tokenize_satb.py musicxml_synth/synth_00000.musicxml
```

### 2. Build full dataset (on RunPod CPU instance)

```sh
# On a fresh RunPod CPU pod (~$0.10/hr):
git clone <your-repo> /workspace/choir-omr
cd /workspace/choir-omr/pipeline
COUNT=30000 AUGMENTS=5 bash runpod_setup.sh generate
```
~3 hours, costs ~$0.30, produces ~150k training pairs in `/workspace/dataset/`.

### 3. Train (on RunPod A100 GPU instance)

```sh
# Attach the same volume from step 2
cd /workspace/choir-omr/pipeline
BATCH=8 EPOCHS=50 bash runpod_setup.sh train
```
~50 hours on A100 (~$80), produces `/workspace/checkpoints/ckpt_epoch_*.pt`.

### 4. Download model & deploy

```sh
# From your Mac:
scp -i ~/.ssh/runpod root@<pod-ip>:/workspace/checkpoints/ckpt_epoch_049.pt \
    ~/Desktop/choir/omr-server/
```

Then wire into `server.py` to use the new model for `/recognize`.

## Cost / time projections

| Phase | Cost | Wall clock |
|-------|------|------------|
| Code (you) | €0 | 1–2 weeks part-time |
| Dataset generation (RunPod CPU) | ~€0.50 | ~3 hours unattended |
| Training (RunPod A100) | ~€80–120 | ~2–3 days unattended |
| Iteration (3 rounds × ~€20) | ~€60 | ~3 days |
| **Total** | **~€150–200** | **~3 weeks part-time** |

## What this gives you

- Custom 80M-param OMR model trained specifically on SATB choir music
- Handles closed score (2 staves) and traditional layouts
- Lyrics included (via tokenized labels)
- Runs on your Mac for inference (~10–15 sec/page)
- All weights yours, no per-scan API fees, no internet required at inference time
- Expected accuracy: ~80–85% on phone photos of SATB scores
