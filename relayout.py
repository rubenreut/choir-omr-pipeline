#!/usr/bin/env python3
"""Generate closed-score variants of SATB MusicXMLs.

Closed score = 2 staves:
  Staff 1 (treble clef): Soprano (voice 1, stems up) + Alto (voice 2, stems down)
  Staff 2 (bass clef):   Tenor   (voice 1, stems up) + Bass  (voice 2, stems down)

We read each open-score (4-part) MusicXML from ./musicxml/ and write a closed-score
version to ./musicxml_closed/.

Usage:
    python3 relayout.py [--limit N]
"""
import argparse
import sys
from pathlib import Path

from music21 import converter, stream, layout, clef

SRC_DIR = Path(__file__).parent / "musicxml"
OUT_DIR = Path(__file__).parent / "musicxml_closed"


def to_closed_score(open_score) -> stream.Score:
    """Convert open-score (4 parts) → closed-score (2 parts × 2 voices each)."""
    parts = list(open_score.parts)
    if len(parts) != 4:
        raise ValueError(f"Expected 4 parts, got {len(parts)}")

    soprano, alto, tenor, bass = parts

    # Build new score with 2 parts: upper (S + A on treble), lower (T + B on bass)
    new = stream.Score()
    new.metadata = open_score.metadata

    upper = stream.Part()
    upper.partName = "Soprano/Alto"
    upper.insert(0, clef.TrebleClef())

    lower = stream.Part()
    lower.partName = "Tenor/Bass"
    lower.insert(0, clef.BassClef())

    # Merge soprano + alto into upper, tenor + bass into lower.
    # music21's voicesToParts (and its inverse) handles voice splitting in MusicXML.
    # Simplest: copy soprano's measures into upper as voice 1, alto's as voice 2, etc.
    soprano_measures = list(soprano.getElementsByClass("Measure"))
    alto_measures = list(alto.getElementsByClass("Measure"))
    tenor_measures = list(tenor.getElementsByClass("Measure"))
    bass_measures = list(bass.getElementsByClass("Measure"))

    n_measures = min(len(soprano_measures), len(alto_measures), len(tenor_measures), len(bass_measures))

    for i in range(n_measures):
        # Upper staff: S as voice 1, A as voice 2
        up_m = stream.Measure(number=i + 1)
        s_voice = stream.Voice(id="1")
        for n in soprano_measures[i].notesAndRests:
            s_voice.append(n)
        a_voice = stream.Voice(id="2")
        for n in alto_measures[i].notesAndRests:
            a_voice.append(n)
        up_m.insert(0, s_voice)
        up_m.insert(0, a_voice)
        upper.append(up_m)

        # Lower staff: T as voice 1, B as voice 2
        lo_m = stream.Measure(number=i + 1)
        t_voice = stream.Voice(id="1")
        for n in tenor_measures[i].notesAndRests:
            t_voice.append(n)
        b_voice = stream.Voice(id="2")
        for n in bass_measures[i].notesAndRests:
            b_voice.append(n)
        lo_m.insert(0, t_voice)
        lo_m.insert(0, b_voice)
        lower.append(lo_m)

    new.insert(0, upper)
    new.insert(0, lower)

    # Group the two staves with a bracket
    sg = layout.StaffGroup([upper, lower], name="Choir", abbreviation="Ch.", symbol="bracket")
    new.insert(0, sg)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not SRC_DIR.exists():
        sys.exit(f"No source dir: {SRC_DIR}. Run extract_music21.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(SRC_DIR.glob("*.musicxml"))
    if args.limit:
        inputs = inputs[:args.limit]

    success, failed = 0, 0
    for i, src in enumerate(inputs, start=1):
        try:
            open_score = converter.parse(str(src))
            closed = to_closed_score(open_score)
            out = OUT_DIR / src.name
            closed.write("musicxml", fp=str(out))
            print(f"[relayout] [{i}/{len(inputs)}] {src.stem} ok", flush=True)
            success += 1
        except Exception as e:
            print(f"[relayout] FAIL {src.stem}: {e}", flush=True)
            failed += 1

    print(f"\n[relayout] done. {success} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
