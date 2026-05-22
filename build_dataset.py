#!/usr/bin/env python3
"""End-to-end dataset build script.

Generates synthetic SATB MusicXMLs, renders to PNGs, augments, tokenizes,
and writes a training manifest. Designed to run on a single instance (locally or in cloud).

Usage:
    python3 build_dataset.py --count 30000 --out ./dataset/
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import random
from pathlib import Path

# Local imports
import synthesize
import tokenize_satb

PIL_QUALITY = 85


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-1000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000, help="Number of unique synthetic pieces")
    ap.add_argument("--augments", type=int, default=5, help="Augmented variants per page")
    ap.add_argument("--out", type=Path, default=Path("./dataset"), help="Output root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--musescore", default="mscore", help="MuseScore CLI binary")
    args = ap.parse_args()

    random.seed(args.seed)
    out = args.out
    musicxml_dir = out / "musicxml"
    images_dir = out / "images"
    augmented_dir = out / "augmented"
    musicxml_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Generate MusicXML ---
    print(f"[build] generating {args.count} MusicXMLs...", flush=True)
    for i in range(args.count):
        score = synthesize.synthesize_one(seed=args.seed + i)
        path = musicxml_dir / f"piece_{i:06d}.musicxml"
        score.write("musicxml", fp=str(path))
        if (i + 1) % 100 == 0:
            print(f"[build] generated {i + 1}/{args.count}", flush=True)

    # --- 2. Render to PNG via MuseScore ---
    print(f"[build] rendering to PNG via MuseScore...", flush=True)
    for i, mx in enumerate(sorted(musicxml_dir.glob("*.musicxml"))):
        piece_id = mx.stem
        piece_dir = images_dir / piece_id
        piece_dir.mkdir(parents=True, exist_ok=True)
        run([args.musescore, "-o", str(piece_dir / f"{piece_id}.png"), str(mx)])
        if (i + 1) % 50 == 0:
            print(f"[build] rendered {i + 1}", flush=True)

    # --- 3. Augment + convert to WebP ---
    print(f"[build] augmenting...", flush=True)
    from PIL import Image
    # Reuse augment.py logic
    import importlib.util
    spec = importlib.util.spec_from_file_location("augment", Path(__file__).parent / "augment.py")
    augment_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augment_mod)

    for i, piece_dir in enumerate(sorted(p for p in images_dir.iterdir() if p.is_dir())):
        pages = sorted(piece_dir.glob("*.png"))
        out_piece = augmented_dir / piece_dir.name
        for page in pages:
            augment_mod.augment_one(page, args.augments, out_piece)
        if (i + 1) % 50 == 0:
            print(f"[build] augmented {i + 1}", flush=True)

    # --- 4. Tokenize + build manifest ---
    print(f"[build] tokenizing + building manifest...", flush=True)
    from music21 import converter
    all_token_seqs: list[list[str]] = []
    entries_raw: list[tuple[str, list[str]]] = []
    for mx in sorted(musicxml_dir.glob("*.musicxml")):
        try:
            score = converter.parse(str(mx))
            tokens = tokenize_satb.encode_score(score)
            piece_id = mx.stem
            all_token_seqs.append(tokens)
            aug_dir = augmented_dir / piece_id
            for aug_img in sorted(aug_dir.glob("*.webp")):
                rel = aug_img.relative_to(out)
                entries_raw.append((str(rel), tokens))
        except Exception as e:
            print(f"[build] skip {mx.stem}: {e}", flush=True)

    # Build vocab and convert tokens → ids
    vocab = tokenize_satb.build_vocab(all_token_seqs)
    print(f"[build] vocab size: {len(vocab)}", flush=True)
    tokenize_satb.save_vocab(vocab, out / "vocab.json")

    manifest = []
    unk_id = vocab["<unk>"]
    for rel, tokens in entries_raw:
        ids = [vocab.get(t, unk_id) for t in tokens]
        manifest.append({"image": rel, "tokens": ids})
    (out / "manifest.json").write_text(json.dumps(manifest))

    print(f"\n[build] DONE", flush=True)
    print(f"  MusicXML files:  {len(list(musicxml_dir.glob('*.musicxml')))}", flush=True)
    print(f"  Augmented pairs: {len(manifest)}", flush=True)
    print(f"  Vocab size:      {len(vocab)}", flush=True)
    print(f"  Output dir:      {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
