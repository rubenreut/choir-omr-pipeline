#!/usr/bin/env python3
"""Pull SATB scores from the music21 corpus and save as MusicXML files.

For now: focuses on Bach four-part chorales (canonical SATB).
Outputs to ./musicxml/<piece_id>.musicxml

Usage:
    python3 extract_music21.py [--limit N]
        --limit N : only extract N pieces (default: all)
"""
import argparse
import os
import sys
from pathlib import Path

from music21 import corpus

OUT_DIR = Path(__file__).parent / "musicxml"


def is_satb(score) -> bool:
    """Heuristic: must have exactly 4 parts with SATB-like names."""
    parts = score.parts
    if len(parts) != 4:
        return False
    names = [(p.partName or "").lower() for p in parts]
    expected = {"soprano", "alto", "tenor", "bass"}
    return all(any(exp in n for n in names) for exp in expected)


def extract_bach_chorales(limit: int | None = None) -> list[Path]:
    """Extract Bach chorales from music21 corpus → MusicXML files. Returns paths written."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Bach chorales are stored under composer="bach". The corpus contains many BWV files;
    # we filter to four-voice chorales.
    print(f"[extract] scanning music21 corpus for Bach chorales...", flush=True)
    bach_files = corpus.getComposer("bach")
    print(f"[extract] found {len(bach_files)} Bach files in corpus", flush=True)

    written: list[Path] = []
    skipped = 0
    for i, path in enumerate(bach_files):
        if limit and len(written) >= limit:
            break
        try:
            score = corpus.parse(path)
        except Exception as e:
            print(f"[extract] parse fail {path}: {e}", flush=True)
            skipped += 1
            continue

        if not is_satb(score):
            skipped += 1
            continue

        # Use the file's basename (e.g. 'bwv66.6') as id
        piece_id = Path(str(path)).stem.replace(" ", "_")
        out_path = OUT_DIR / f"{piece_id}.musicxml"
        try:
            score.write("musicxml", fp=str(out_path))
            written.append(out_path)
            print(f"[extract] [{len(written)}] {piece_id}", flush=True)
        except Exception as e:
            print(f"[extract] write fail {piece_id}: {e}", flush=True)
            skipped += 1

    print(f"\n[extract] done. Wrote {len(written)} SATB MusicXMLs to {OUT_DIR}/", flush=True)
    print(f"[extract] skipped {skipped} (non-SATB or errors)", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to N pieces (default: all)")
    args = ap.parse_args()

    written = extract_bach_chorales(limit=args.limit)
    if not written:
        sys.exit("No SATB pieces extracted")
    print(f"\nExample: {written[0]}")


if __name__ == "__main__":
    main()
