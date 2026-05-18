"""Tests for MidiToMusicxmlConverter."""
from fractions import Fraction
from pathlib import Path

import music21 as m21
import pretty_midi
import pytest

from models.base.midi_to_musicxml import MidiToMusicxmlConverter


def _midi_with_n_quarter_notes(n: int) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for i in range(n):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5)
        )
    pm.instruments.append(inst)
    return pm


def test_to_melody_only_returns_score_with_notes():
    pm = _midi_with_n_quarter_notes(4)
    score = MidiToMusicxmlConverter.to_melody_only(pm)
    assert isinstance(score, m21.stream.Score)
    notes = list(score.recurse().getElementsByClass(m21.note.Note))
    assert len(notes) == 4


def test_to_melody_only_does_not_mutate_input_midi():
    """The original PrettyMIDI must be untouched — save_midi reads it
    later and must see the original timings."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    raw_start, raw_end = 0.07, 0.13
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=raw_start, end=raw_end)
    )
    pm.instruments.append(inst)

    _ = MidiToMusicxmlConverter.to_melody_only(pm)

    assert pm.instruments[0].notes[0].start == raw_start
    assert pm.instruments[0].notes[0].end == raw_end


def test_to_melody_only_yields_well_aligned_measures():
    """Every Measure must have duration == 4.0 quarters (4/4 time
    signature) regardless of irregular note timings."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for i in range(16):
        irregularity = (i % 3 - 1) * 0.005
        start = i * 0.5 + irregularity
        end = (i + 1) * 0.5 + irregularity
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=60 + (i % 8), start=start, end=end)
        )
    pm.instruments.append(inst)

    score = MidiToMusicxmlConverter.to_melody_only(pm)
    measures = list(score.recurse().getElementsByClass(m21.stream.Measure))
    assert len(measures) >= 1
    for m in measures:
        assert abs(float(m.duration.quarterLength) - 4.0) < 1e-6, (
            f"Measure {m.number} has duration {float(m.duration.quarterLength)}"
        )


def test_to_melody_only_renders_triplet_quarter_with_bracket():
    """A quarter containing notes at triplet positions (0, 8/24, 16/24)
    must be bracketed as a full 8th-triplet group of three slots."""
    # 120 BPM → quarter = 0.5 sec → 8th-triplet = 1/6 sec.
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    triplet_dur = 0.5 / 3
    for i in range(3):
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=60 + i,
                start=i * triplet_dur,
                end=(i + 1) * triplet_dur,
            )
        )
    pm.instruments.append(inst)

    score = MidiToMusicxmlConverter.to_melody_only(pm)
    notes = list(score.recurse().getElementsByClass(m21.note.Note))
    # Three triplet members in the first quarter.
    triplet_notes = [n for n in notes if n.duration.tuplets]
    assert len(triplet_notes) >= 3, (
        f"Expected ≥3 triplet-bracketed notes, got {len(triplet_notes)}"
    )


def test_to_melody_only_uses_power_of_2_for_sixteenths():
    """Plain 16ths (positions on the 16-grid only) must NOT be bracketed
    as triplets — power-of-2 mode."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Four 16ths inside one quarter.
    for i in range(4):
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=60 + i,
                start=i * 0.125,
                end=(i + 1) * 0.125,
            )
        )
    pm.instruments.append(inst)

    score = MidiToMusicxmlConverter.to_melody_only(pm)
    notes = list(score.recurse().getElementsByClass(m21.note.Note))
    triplet_notes = [n for n in notes if n.duration.tuplets]
    assert not triplet_notes, "16th-grid notes must not be tripleted"


def test_to_melody_only_merges_tied_chains_to_clean_durations():
    """Four tied quarters of the same pitch should collapse into ONE
    whole note after the cosmetic merge pass."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0)  # whole at 120bpm
    )
    pm.instruments.append(inst)

    score = MidiToMusicxmlConverter.to_melody_only(pm)
    notes = list(score.recurse().getElementsByClass(m21.note.Note))
    assert len(notes) == 1, f"Expected 1 merged whole-note, got {len(notes)}"
    assert abs(float(notes[0].duration.quarterLength) - 4.0) < 1e-9


def test_to_melody_only_writable_as_musicxml(tmp_path):
    pm = _midi_with_n_quarter_notes(4)
    score = MidiToMusicxmlConverter.to_melody_only(pm)
    out = tmp_path / "out.musicxml"
    score.write("musicxml", fp=str(out))
    assert out.is_file()
    # Round-trip parse to confirm it's valid MusicXML.
    parsed = m21.converter.parse(str(out))
    assert len(list(parsed.recurse().getElementsByClass(m21.note.Note))) == 4


def test_to_with_chords_tiles_cyclically():
    """2-bar theme with 2 chords → 4-bar output (input_bars=2, output_bars=2)
    tiles the 2 chords twice → 4 ChordSymbol elements total."""
    pm = _midi_with_n_quarter_notes(16)  # 4 bars worth of quarters
    chord_symbols = [
        (0.0, m21.harmony.ChordSymbol("C")),
        (4.0, m21.harmony.ChordSymbol("G7")),
    ]
    score = MidiToMusicxmlConverter.to_with_chords(
        pm, chord_symbols, input_bars=2, output_bars=2
    )
    chords = list(score.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert len(chords) == 4
    # cycles: 0 → [0, 4]; 1 → [8, 12]. cycle_length = 2*4 = 8 q.
    offsets = sorted(float(cs.getOffsetInHierarchy(score)) for cs in chords)
    assert offsets == [0.0, 4.0, 8.0, 12.0]


def test_to_with_chords_no_symbols_returns_melody_only():
    pm = _midi_with_n_quarter_notes(8)
    score = MidiToMusicxmlConverter.to_with_chords(
        pm, theme_chord_symbols=[], input_bars=2, output_bars=2
    )
    assert len(list(score.recurse().getElementsByClass(m21.harmony.ChordSymbol))) == 0
    assert len(list(score.recurse().getElementsByClass(m21.note.Note))) == 8


def test_to_with_chords_clips_overflow():
    """Last cycle should not place chords past total bars."""
    pm = _midi_with_n_quarter_notes(12)  # 3 bars worth of quarters
    chord_symbols = [
        (0.0, m21.harmony.ChordSymbol("C")),
        (4.0, m21.harmony.ChordSymbol("G7")),
        (6.0, m21.harmony.ChordSymbol("F")),
    ]
    # input_bars=2, output_bars=1 → total = 3 bars = 12 q. cycle_length = 8 q.
    # Cycle 0: 0(C), 4(G7), 6(F). Cycle 1: 8(C), 12(clip G7), 14(clip F).
    # → 4 chords total.
    score = MidiToMusicxmlConverter.to_with_chords(
        pm, chord_symbols, input_bars=2, output_bars=1
    )
    chords = list(score.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert len(chords) == 4


def test_to_with_chords_survives_musicxml_roundtrip(tmp_path):
    """ChordSymbols must round-trip through write('musicxml') →
    converter.parse — only true if we put them inside Measures."""
    pm = _midi_with_n_quarter_notes(8)
    chord_symbols = [
        (0.0, m21.harmony.ChordSymbol("C")),
        (4.0, m21.harmony.ChordSymbol("G7")),
    ]
    score = MidiToMusicxmlConverter.to_with_chords(
        pm, chord_symbols, input_bars=2, output_bars=0
    )
    out = tmp_path / "out.musicxml"
    score.write("musicxml", fp=str(out))
    parsed = m21.converter.parse(str(out))
    re_chords = list(parsed.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert len(re_chords) == 2
