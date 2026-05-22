#!/usr/bin/env python3
"""Generate random but musically-plausible SATB pieces with proper voice conventions.

Outputs CLOSED-SCORE MusicXML directly (2 staves: S+A treble, T+B bass).
Voices are correctly structured for the model to learn standard SATB conventions:
  Top staff:    voice 1 = Soprano (stems up), voice 2 = Alto (stems down)
  Bottom staff: voice 1 = Tenor  (stems up), voice 2 = Bass (stems down)

When two voices share the same note, they share a notehead (chord, no doubling).
When one voice rests, the other is still rendered with its proper stem direction.

Usage:
    python3 synthesize.py [--count N] [--seed S]
"""
import argparse
import random
import sys
from pathlib import Path

from music21 import (
    chord, clef, duration, instrument, key, layout, meter, note, pitch, stream
)

OUT_DIR = Path(__file__).parent / "musicxml_synth"

# Voice ranges (MIDI numbers — typical choral ranges)
SOPRANO_RANGE = (60, 79)   # C4–G5
ALTO_RANGE    = (55, 74)   # G3–D5
TENOR_RANGE   = (48, 67)   # C3–G4
BASS_RANGE    = (40, 62)   # E2–D4

# Lyric vocabulary (mix of plausible Latin/English/German syllables)
SYLLABLES = [
    # Latin
    "Glo", "ri", "a", "in", "ex", "cel", "sis", "De", "o",
    "Ky", "ri", "e", "e", "le", "i", "son",
    "Sanc", "tus", "Be", "ne", "dic", "tus",
    "A", "ve", "Ma", "ri", "a", "gra", "ti", "a", "ple", "na",
    "Pa", "ter", "nos", "ter",
    "A", "men", "Al", "le", "lu", "ia",
    # English
    "Lord", "have", "mer", "cy", "praise", "ye", "the", "name",
    "Ho", "ly", "ho", "ly", "ho", "ly", "is", "the", "Lord",
    "sing", "we", "to", "our", "God", "a", "bove",
    # German
    "Hei", "lig", "ist", "der", "Herr", "Got", "tes", "Sohn",
    "Ge", "lo", "bet", "sei", "der", "Herr",
]


def pick_in_scale(pitch_class_set: list[int], range_min: int, range_max: int,
                  prev_midi: int | None = None,
                  forbidden: set[int] | None = None) -> int:
    """Pick a random MIDI note within range that's in the given pitch-class set,
    preferring stepwise motion from prev_midi if provided.
    Excludes any midi values in `forbidden` (used to avoid second intervals between voices)."""
    candidates = [m for m in range(range_min, range_max + 1)
                  if m % 12 in pitch_class_set and (not forbidden or m not in forbidden)]
    if not candidates:
        # Relax the forbidden constraint if it left us with no candidates
        candidates = [m for m in range(range_min, range_max + 1) if m % 12 in pitch_class_set]
        if not candidates:
            return (range_min + range_max) // 2

    if prev_midi is not None:
        # Prefer notes within an octave of prev, weight by proximity
        weights = []
        for c in candidates:
            d = abs(c - prev_midi)
            if d == 0:
                w = 0.5
            elif d <= 2:
                w = 4.0      # strong preference for stepwise
            elif d <= 4:
                w = 2.0
            elif d <= 7:
                w = 1.0
            elif d <= 12:
                w = 0.4
            else:
                w = 0.05
            weights.append(w)
        return random.choices(candidates, weights=weights, k=1)[0]
    return random.choice(candidates)


def generate_rhythm(total_quarters: float, max_subdivision: int = 4) -> list[float]:
    """Generate a random rhythm pattern summing to total_quarters.
    Durations are expressed in quarter-note units."""
    # Allowable durations (in quarters): whole, half, dotted-quarter, quarter, eighth
    pool = [4.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    rhythm: list[float] = []
    remaining = total_quarters
    while remaining > 0:
        candidates = [d for d in pool if d <= remaining + 1e-6]
        if not candidates:
            rhythm.append(remaining)
            break
        d = random.choice(candidates)
        rhythm.append(d)
        remaining -= d
    return rhythm


def generate_voice_line(scale_pcs: list[int], range_lo: int, range_hi: int,
                        n_measures: int, beats_per_measure: float,
                        starting_midi: int | None = None,
                        rhythm_template: list[list[float]] | None = None,
                        rest_probability: float = 0.04,
                        upper_bound_voice: list[list[tuple[int | None, float]]] | None = None
                        ) -> list[list[tuple[int | None, float]]]:
    """Generate a voice line as list of measures, each a list of (midi_or_None, duration_in_quarters).
    If rhythm_template is provided, use that exact rhythm for each measure (so all voices align).
    If upper_bound_voice is provided (must share rhythm), each generated note's pitch is capped
    at the upper_bound_voice's note at the same position — enforces S ≥ A ≥ T ≥ B."""
    measures: list[list[tuple[int | None, float]]] = []
    prev = starting_midi or (range_lo + range_hi) // 2

    for m in range(n_measures):
        rhythm = rhythm_template[m] if rhythm_template else generate_rhythm(beats_per_measure)
        notes_this_measure: list[tuple[int | None, float]] = []
        for n_idx, dur in enumerate(rhythm):
            if random.random() < rest_probability:
                notes_this_measure.append((None, dur))
                continue

            # Cap range_hi by the corresponding upper-voice note (if rhythm-aligned)
            effective_hi = range_hi
            forbidden_pitches: set[int] = set()
            if upper_bound_voice and m < len(upper_bound_voice) and n_idx < len(upper_bound_voice[m]):
                upper = upper_bound_voice[m][n_idx][0]
                if upper is not None:
                    # Allow equal pitch (unison) OR at least a third (3 semitones) below.
                    # Forbid pitches that would create a "second" interval (1 or 2 semitones below).
                    effective_hi = min(range_hi, upper)
                    forbidden_pitches = {upper - 1, upper - 2}
                    if effective_hi < range_lo:
                        notes_this_measure.append((None, dur))
                        continue

            midi = pick_in_scale(scale_pcs, range_lo, effective_hi, prev,
                                  forbidden=forbidden_pitches)
            notes_this_measure.append((midi, dur))
            prev = midi
        measures.append(notes_this_measure)
    return measures


def random_lyric_syllables(n_syllables: int) -> list[tuple[str, str]]:
    """Generate n_syllables (text, syllabic_type) tuples forming plausible word boundaries."""
    result: list[tuple[str, str]] = []
    i = 0
    while i < n_syllables:
        # Word length: 1-4 syllables
        word_len = random.choices([1, 2, 3, 4], weights=[3, 5, 3, 1])[0]
        word_len = min(word_len, n_syllables - i)
        for j in range(word_len):
            syl = random.choice(SYLLABLES)
            if word_len == 1:
                syl_type = "single"
            elif j == 0:
                syl_type = "begin"
            elif j == word_len - 1:
                syl_type = "end"
            else:
                syl_type = "middle"
            result.append((syl, syl_type))
        i += word_len
    return result


def add_lyrics_to_voice(voice_measures: list[list[tuple[int | None, float]]],
                        verse_number: int = 1) -> dict[int, list[tuple[int, str, str]]]:
    """Returns dict {measure_idx: [(note_idx, text, syllabic), ...]} for non-rest notes."""
    # Count non-rest notes
    note_positions: list[tuple[int, int]] = []  # (measure_idx, note_idx_within_measure)
    for m_idx, m in enumerate(voice_measures):
        for n_idx, (midi, _) in enumerate(m):
            if midi is not None:
                note_positions.append((m_idx, n_idx))

    syllables = random_lyric_syllables(len(note_positions))
    out: dict[int, list[tuple[int, str, str]]] = {}
    for (m_idx, n_idx), (syl, syl_type) in zip(note_positions, syllables):
        out.setdefault(m_idx, []).append((n_idx, syl, syl_type))
    return out


def make_note(midi: int, dur: float, stem_dir: str | None = None,
              midi_to_pitch: dict[int, pitch.Pitch] | None = None) -> note.Note:
    """Build a Note with proper key-aware pitch spelling so it doesn't render as an accidental.
    stem_dir of None lets the renderer auto-pick based on staff position (preferred)."""
    if midi_to_pitch and midi in midi_to_pitch:
        n = note.Note(midi_to_pitch[midi], quarterLength=dur)
    else:
        n = note.Note(midi, quarterLength=dur)
    if stem_dir is not None:
        n.stemDirection = stem_dir
    # Suppress explicit accidental display — let the key signature handle it.
    if n.pitch.accidental is not None:
        n.pitch.accidental.displayStatus = False
    return n


def build_voice_part(voice_measures: list[list[tuple[int | None, float]]],
                     part_name: str, clef_obj,
                     voice_id: str,
                     staff_default_stem: str,  # "up" for treble (S+A), "down" for bass (T+B)
                     lyrics_by_measure: dict[int, list[tuple[int, str, str]]] | None = None,
                     additional_voice_measures: list[list[tuple[int | None, float]]] | None = None,
                     additional_voice_id: str = "2",
                     additional_lyrics_by_measure: dict[int, list[tuple[int, str, str]]] | None = None,
                     time_sig: meter.TimeSignature | None = None,
                     key_sig: key.KeySignature | None = None,
                     midi_to_pitch: dict[int, pitch.Pitch] | None = None,
                     engraving_style: str = "merged",  # "merged" or "traditional"
                     upper_voice_stem: str = "up",
                     lower_voice_stem: str = "down",
                     ) -> stream.Part:
    """Build a Part with two voices on one staff (closed-score style).

    Engraving rules implemented:
      - When both voices have a note at the same beat with same rhythm:
            same pitch (unison)  → single shared notehead with two opposing stems
            different pitches    → CHORD: one stem (staff_default), shared beam,
                                   both noteheads on the same stem
      - When one voice rests and the other has a note → single note with staff_default stem
      - When voices have different rhythms (polyphonic):
            → fall back to two separate voices with separate stems
    """
    part = stream.Part()
    part.partName = part_name
    part.insert(0, clef_obj)
    if key_sig:
        part.insert(0, key_sig)
    if time_sig:
        part.insert(0, time_sig)

    n_measures = len(voice_measures)
    for m_idx in range(n_measures):
        measure = stream.Measure(number=m_idx + 1)
        v1_measure = voice_measures[m_idx]
        v2_measure = additional_voice_measures[m_idx] if additional_voice_measures else []

        # Decide whether rhythms align (so we can use chord-merging)
        rhythms_aligned = (
            additional_voice_measures
            and len(v1_measure) == len(v2_measure)
            and all(v1_measure[i][1] == v2_measure[i][1] for i in range(len(v1_measure)))
        )

        # Determine if every beat is either (both notes) or (both rests) — eligible for chord-merge.
        # If ANY beat has one note + one rest, we MUST keep separate voices so the resting voice's
        # rest symbol is visible to the reader.
        chord_mergeable = rhythms_aligned and all(
            (v1_measure[i][0] is None) == (v2_measure[i][0] is None)
            for i in range(len(v1_measure))
        )

        if chord_mergeable and engraving_style == "merged":
            # CHORD-MERGE PATH: single voice on the staff, chord at each beat
            v = stream.Voice(id=voice_id)
            for n_idx, ((m1, dur1), (m2, _)) in enumerate(zip(v1_measure, v2_measure)):
                if m1 is None and m2 is None:
                    el = note.Rest(quarterLength=dur1)
                elif m1 == m2:
                    # Unison: single notehead with staff-default stem
                    el = make_note(m1, dur1, staff_default_stem, midi_to_pitch)
                else:
                    # Different pitches → CHORD with shared stem
                    n_top = make_note(max(m1, m2), dur1, staff_default_stem, midi_to_pitch)
                    n_bot = make_note(min(m1, m2), dur1, staff_default_stem, midi_to_pitch)
                    el = chord.Chord([n_top, n_bot], quarterLength=dur1)
                    el.stemDirection = staff_default_stem

                if lyrics_by_measure:
                    for (target_n_idx, text, syl_type) in lyrics_by_measure.get(m_idx, []):
                        if target_n_idx == n_idx and not isinstance(el, note.Rest):
                            el.addLyric(text)
                            if el.lyrics:
                                el.lyrics[-1].syllabic = syl_type
                v.append(el)
            measure.insert(0, v)
        else:
            # SEPARATE-VOICES PATH:
            # Traditional style → upper voice stems up, lower voice stems down (opposing).
            # Otherwise (mixed-rest-and-note case) → both use staff_default_stem.
            use_opposing = (engraving_style == "traditional")
            v1_stem = upper_voice_stem if use_opposing else staff_default_stem
            v2_stem = lower_voice_stem if use_opposing else staff_default_stem

            v1 = stream.Voice(id=voice_id)
            for n_idx, (midi, dur) in enumerate(v1_measure):
                if midi is None:
                    el = note.Rest(quarterLength=dur)
                else:
                    el = make_note(midi, dur, v1_stem, midi_to_pitch)
                if lyrics_by_measure:
                    for (target_n_idx, text, syl_type) in lyrics_by_measure.get(m_idx, []):
                        if target_n_idx == n_idx and midi is not None:
                            el.addLyric(text)
                            if el.lyrics:
                                el.lyrics[-1].syllabic = syl_type
                v1.append(el)
            measure.insert(0, v1)

            if additional_voice_measures:
                v2 = stream.Voice(id=additional_voice_id)
                for n_idx, (midi, dur) in enumerate(v2_measure):
                    if midi is None:
                        el = note.Rest(quarterLength=dur)
                    else:
                        el = make_note(midi, dur, v2_stem, midi_to_pitch)
                    if additional_lyrics_by_measure:
                        for (target_n_idx, text, syl_type) in additional_lyrics_by_measure.get(m_idx, []):
                            if target_n_idx == n_idx and midi is not None:
                                el.addLyric(text)
                                if el.lyrics:
                                    el.lyrics[-1].syllabic = syl_type
                    v2.append(el)
                measure.insert(0, v2)

        part.append(measure)

    return part


def synthesize_one(seed: int | None = None) -> stream.Score:
    """Generate one random SATB piece in closed-score format."""
    if seed is not None:
        random.seed(seed)

    # Random params
    n_measures = random.randint(4, 16)
    time_choice = random.choice([(4, 4), (4, 4), (3, 4), (6, 8), (2, 2)])
    beats_per_measure = time_choice[0] * (4 / time_choice[1])  # in quarter notes
    time_sig = meter.TimeSignature(f"{time_choice[0]}/{time_choice[1]}")

    # Random key — use music21.key.Key so the proper pitch spellings are known
    mode = random.choice(["major", "minor"])
    tonic_letter = random.choice(["C", "D", "E", "F", "G", "A", "B"])
    accidental = random.choice(["", "", "", "#", "-"])  # mostly natural keys
    tonic_name = f"{tonic_letter}{accidental}"
    try:
        key_obj = key.Key(tonic_name, mode)
    except Exception:
        key_obj = key.Key("C", mode)
    fifths = key_obj.sharps
    key_sig = key.KeySignature(fifths)
    # Build a midi -> properly-spelled-pitch map for this key by walking the scale
    scale_pitches = key_obj.getPitches("C2", "C7")  # pitches from C2..C7 in this key
    midi_to_pitch: dict[int, pitch.Pitch] = {}
    for p in scale_pitches:
        midi_to_pitch[p.midi] = p
    scale_pcs = sorted({p.midi % 12 for p in scale_pitches})

    # Pick engraving style first — affects rest behavior.
    engraving_style = random.choice(["merged", "merged", "traditional"])

    # Always homophonic (all voices share rhythm) so chord-merge works per-beat for shared beams.
    soprano_rhythms = [generate_rhythm(beats_per_measure) for _ in range(n_measures)]
    rest_prob = 0.02

    soprano = generate_voice_line(scale_pcs, *SOPRANO_RANGE, n_measures, beats_per_measure,
                                   rhythm_template=soprano_rhythms, rest_probability=rest_prob)
    alto    = generate_voice_line(scale_pcs, *ALTO_RANGE, n_measures, beats_per_measure,
                                   rhythm_template=soprano_rhythms, rest_probability=rest_prob,
                                   upper_bound_voice=soprano)
    tenor   = generate_voice_line(scale_pcs, *TENOR_RANGE, n_measures, beats_per_measure,
                                   rhythm_template=soprano_rhythms, rest_probability=rest_prob,
                                   upper_bound_voice=alto)
    bass    = generate_voice_line(scale_pcs, *BASS_RANGE, n_measures, beats_per_measure,
                                   rhythm_template=soprano_rhythms, rest_probability=rest_prob,
                                   upper_bound_voice=tenor)

    # In "merged" style, force rests to be shared between voices on the same staff
    # (so chord-merge fires for every beat and stems/beams are consistently shared).
    if engraving_style == "merged":
        def sync_rests(upper, lower):
            for m_idx in range(len(upper)):
                for n_idx in range(min(len(upper[m_idx]), len(lower[m_idx]))):
                    u_midi, u_dur = upper[m_idx][n_idx]
                    l_midi, l_dur = lower[m_idx][n_idx]
                    if u_midi is None and l_midi is not None:
                        lower[m_idx][n_idx] = (None, l_dur)
                    elif l_midi is None and u_midi is not None:
                        upper[m_idx][n_idx] = (None, u_dur)
        sync_rests(soprano, alto)
        sync_rests(tenor, bass)

    # Lyric scenarios — lyrics only go on voice 1 of each staff (otherwise they collide
    # visually below the staff). Common conventions:
    #   "none"        : instrumental, no lyrics
    #   "soprano_only": lyrics on Soprano only (most common for hymn)
    #   "both_staves" : lyrics on Soprano AND Tenor (one per staff, no collision)
    lyric_mode = random.choice(["none", "soprano_only", "soprano_only", "both_staves"])

    sop_lyrics: dict | None = None
    alt_lyrics: dict | None = None
    ten_lyrics: dict | None = None
    bas_lyrics: dict | None = None

    if lyric_mode == "soprano_only":
        sop_lyrics = add_lyrics_to_voice(soprano, verse_number=1)
    elif lyric_mode == "both_staves":
        sop_lyrics = add_lyrics_to_voice(soprano, verse_number=1)
        ten_lyrics = add_lyrics_to_voice(tenor, verse_number=1)

    # Build closed-score: 2 parts each with 2 voices
    treble_part = build_voice_part(
        voice_measures=soprano, part_name="Soprano/Alto", clef_obj=clef.TrebleClef(),
        voice_id="1", staff_default_stem="up", lyrics_by_measure=sop_lyrics,
        additional_voice_measures=alto, additional_voice_id="2",
        additional_lyrics_by_measure=alt_lyrics,
        time_sig=time_sig, key_sig=key_sig, midi_to_pitch=midi_to_pitch,
        engraving_style=engraving_style,
        upper_voice_stem="up", lower_voice_stem="down",
    )
    bass_part = build_voice_part(
        voice_measures=tenor, part_name="Tenor/Bass", clef_obj=clef.BassClef(),
        voice_id="1", staff_default_stem="down", lyrics_by_measure=ten_lyrics,
        additional_voice_measures=bass, additional_voice_id="2",
        additional_lyrics_by_measure=bas_lyrics,
        time_sig=meter.TimeSignature(f"{time_choice[0]}/{time_choice[1]}"),
        key_sig=key.KeySignature(fifths), midi_to_pitch=midi_to_pitch,
        engraving_style=engraving_style,
        upper_voice_stem="up", lower_voice_stem="down",
    )

    score = stream.Score()
    score.insert(0, treble_part)
    score.insert(0, bass_part)
    sg = layout.StaffGroup([treble_part, bass_part], name="Choir", abbreviation="Ch.", symbol="bracket")
    score.insert(0, sg)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5, help="Number of pieces to generate")
    ap.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        seed = (args.seed + i) if args.seed is not None else None
        try:
            score = synthesize_one(seed=seed)
            out = OUT_DIR / f"synth_{i:05d}.musicxml"
            score.write("musicxml", fp=str(out))
            print(f"[synth] [{i+1}/{args.count}] {out.name}", flush=True)
        except Exception as e:
            print(f"[synth] FAIL #{i}: {e}", flush=True)

    print(f"\n[synth] done. Output: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
