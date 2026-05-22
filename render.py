#!/usr/bin/env python3
"""Render each MusicXML in ./musicxml/ to PNG via MuseScore CLI.

For multi-page scores, MuseScore writes <name>-1.png, <name>-2.png, etc.
We rename/collect them under ./images/<piece_id>/page_<n>.png.

Usage:
    python3 render.py [--limit N]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MUSESCORE_BIN = shutil.which("mscore") or "/opt/homebrew/bin/mscore"
SRC_DIR = Path(__file__).parent / "musicxml"
OUT_DIR = Path(__file__).parent / "images"


def render_one(musicxml_path: Path) -> list[Path]:
    """Render one MusicXML to PNGs. Returns list of output PNG paths."""
    piece_id = musicxml_path.stem
    piece_dir = OUT_DIR / piece_id
    piece_dir.mkdir(parents=True, exist_ok=True)

    # MuseScore writes <piece_id>-1.png, <piece_id>-2.png, ... in piece_dir
    out_template = piece_dir / f"{piece_id}.png"
    cmd = [MUSESCORE_BIN, "-o", str(out_template), str(musicxml_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if proc.returncode != 0:
        raise RuntimeError(f"mscore failed for {piece_id}: {proc.stderr[-300:]}")

    # Collect pages, rename to page_1.png, page_2.png, ...
    raw_pages = sorted(piece_dir.glob(f"{piece_id}-*.png"))
    final_pages: list[Path] = []
    for i, p in enumerate(raw_pages, start=1):
        final = piece_dir / f"page_{i}.png"
        p.rename(final)
        final_pages.append(final)
    return final_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not Path(MUSESCORE_BIN).exists():
        sys.exit(f"MuseScore binary not found: {MUSESCORE_BIN}")
    if not SRC_DIR.exists():
        sys.exit(f"No source dir: {SRC_DIR}. Run extract_music21.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(SRC_DIR.glob("*.musicxml"))
    if args.limit:
        inputs = inputs[:args.limit]

    print(f"[render] {len(inputs)} MusicXML files to render", flush=True)

    success, failed = 0, 0
    for i, path in enumerate(inputs, start=1):
        try:
            pages = render_one(path)
            print(f"[render] [{i}/{len(inputs)}] {path.stem}: {len(pages)} page(s)", flush=True)
            success += 1
        except Exception as e:
            print(f"[render] FAIL {path.stem}: {e}", flush=True)
            failed += 1

    print(f"\n[render] done. {success} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
