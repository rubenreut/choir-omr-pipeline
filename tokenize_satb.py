#!/usr/bin/env python3
"""MusicXML <-> token-sequence converter for SATB choir scores.

Token vocabulary (~ 600 fixed tokens + lyric subwords):

  Special:        <bos> <eos> <pad> <unk>
  Structure:      <bar> <bar:double> <bar:final> <staff:treble> <staff:bass>
  Clefs:          <clef:G2> <clef:F4>
  Keys:           <key:-7> ... <key:0> ... <key:+7>
  Time sigs:      <time:4/4> <time:3/4> <time:6/8> <time:2/2> <time:2/4> <time:3/8> <time:9/8> <time:12/8>
  Notes:          <note:pitch_dur>     (e.g. <note:C4_quarter>)  ~ 300 combos
  Chords:         <chord:p1_p2_dur>    (e.g. <chord:G4_E4_quarter>) — for chord-merged SA/TB
  Rests:          <rest:whole> <rest:half> <rest:quarter> <rest:eighth> <rest:16th>
  Modifiers:      <dot> <tie> <slur:start> <slur:end> <fermata>
  Voice markers:  <voice:1> <voice:2>   (used in traditional engraving)
  Lyric:          <lyric:[verse]:[syllabic]:[text]>  (text is BPE-tokenized separately)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable

from music21 import converter, key, meter, note, chord, stream, clef as m21clef


# --- Pitch+duration token table ---
PITCH_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
ACCIDENTALS = ["", "-", "#"]  # natural, flat, sharp
OCTAVES = list(range(2, 7))
DURATION_NAMES = {
    4.0: "whole", 2.0: "half", 1.5: "half.", 1.0: "quarter", 0.75: "quarter.",
    0.5: "eighth", 0.25: "16th", 0.125: "32nd",
}


def dur_name(quarter_length: float) -> str:
    # Snap to nearest known duration
    best = min(DURATION_NAMES.keys(), key=lambda d: abs(d - quarter_length))
    return DURATION_NAMES[best]


def pitch_token(p) -> str:
    """Encode a music21 Pitch as e.g. 'C4', 'C#4', 'Bb3'."""
    step = p.step
    octv = p.octave or 4
    acc = ""
    if p.accidental is not None:
        if p.accidental.alter == 1:
            acc = "#"
        elif p.accidental.alter == -1:
            acc = "b"
    return f"{step}{acc}{octv}"


def encode_score(score: stream.Score) -> list[str]:
    """Convert a music21 Score → flat token sequence."""
    tokens: list[str] = ["<bos>"]

    parts = list(score.parts)
    # We assume closed-score SATB: parts[0] = Soprano/Alto (treble), parts[1] = Tenor/Bass (bass)
    for part_idx, part in enumerate(parts):
        staff_marker = "<staff:treble>" if part_idx == 0 else "<staff:bass>"
        tokens.append(staff_marker)

        # Emit clef + key sig + time sig once (from first measure)
        first_measure = next(part.getElementsByClass("Measure").__iter__(), None)
        if first_measure is not None:
            for c in first_measure.getElementsByClass(m21clef.Clef):
                if isinstance(c, m21clef.TrebleClef):
                    tokens.append("<clef:G2>")
                elif isinstance(c, m21clef.BassClef):
                    tokens.append("<clef:F4>")
            for k in first_measure.getElementsByClass(key.KeySignature):
                tokens.append(f"<key:{k.sharps:+d}>")
            for t in first_measure.getElementsByClass(meter.TimeSignature):
                tokens.append(f"<time:{t.numerator}/{t.denominator}>")

        for measure in part.getElementsByClass("Measure"):
            voices = list(measure.voices) if measure.voices else [measure]
            for v_idx, voice in enumerate(voices):
                if len(voices) > 1:
                    tokens.append(f"<voice:{v_idx + 1}>")
                for el in voice.notesAndRests:
                    if isinstance(el, note.Rest):
                        tokens.append(f"<rest:{dur_name(float(el.quarterLength))}>")
                    elif isinstance(el, chord.Chord):
                        # Top-down list of pitches in the chord
                        pitches_sorted = sorted(el.pitches, key=lambda p: -p.midi)
                        ptoks = "_".join(pitch_token(p) for p in pitches_sorted)
                        tokens.append(f"<chord:{ptoks}_{dur_name(float(el.quarterLength))}>")
                        # Lyrics on chord
                        for lyric_idx, lyr in enumerate(el.lyrics or []):
                            tokens.append(_lyric_token(lyric_idx + 1, lyr))
                    elif isinstance(el, note.Note):
                        tokens.append(f"<note:{pitch_token(el.pitch)}_{dur_name(float(el.quarterLength))}>")
                        for lyric_idx, lyr in enumerate(el.lyrics or []):
                            tokens.append(_lyric_token(lyric_idx + 1, lyr))
            tokens.append("<bar>")

    tokens.append("<eos>")
    return tokens


def _lyric_token(verse_num: int, lyric) -> str:
    """Encode a music21 Lyric object as a token."""
    text = (lyric.text or "").replace(" ", "_")
    syl = lyric.syllabic or "single"
    return f"<lyric:{verse_num}:{syl}:{text}>"


# ---- Vocab builder ----

def build_vocab(token_sequences: Iterable[list[str]]) -> dict[str, int]:
    """Walk all tokens and assign each unique token an id."""
    vocab: dict[str, int] = {
        "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    }
    for seq in token_sequences:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())


# ---- CLI ----

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("musicxml", type=Path, help="A single MusicXML file to encode")
    ap.add_argument("--print", action="store_true", help="Print all tokens")
    args = ap.parse_args()

    score = converter.parse(str(args.musicxml))
    tokens = encode_score(score)
    print(f"Total tokens: {len(tokens)}")
    if args.print:
        for t in tokens:
            print(t)
    else:
        # Sample first 60
        for t in tokens[:60]:
            print(t)
        print("...")


if __name__ == "__main__":
    main()
