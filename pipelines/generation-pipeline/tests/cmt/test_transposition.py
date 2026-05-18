"""Tests for transposition.py — keys, intervals, in-place midi shifts."""
import music21 as m21
import pytest

from models.cmt.transposition import (
    analyze_key,
    transpose_to_target,
    transpose_midi_back,
)


def _build_one_bar_stream(notes, time_sig="4/4"):
    s = m21.stream.Stream()
    s.append(m21.meter.TimeSignature(time_sig))
    for pitch_name, ql in notes:
        s.append(m21.note.Note(pitch_name, quarterLength=ql))
    return s


# ---------- analyze_key ----------

def test_analyze_key_returns_c_major_for_c_scale():
    s = _build_one_bar_stream([("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)])
    key = analyze_key(s)
    assert isinstance(key, m21.key.Key)
    assert key.tonic.name == "C"
    assert key.mode == "major"


def test_analyze_key_rejects_non_4_4_time_signature():
    s = _build_one_bar_stream([("C4", 1), ("D4", 1), ("E4", 0.5)], time_sig="3/4")
    with pytest.raises(ValueError, match="4/4"):
        analyze_key(s)


def test_analyze_key_returns_a_minor_for_a_minor_scale():
    # Different key + different mode — guards against a hardcoded-return stub
    # passing the C-major test.
    s = _build_one_bar_stream([("A4", 1), ("B4", 1), ("C5", 1), ("D5", 1), ("E5", 1)])
    key = analyze_key(s)
    assert key.tonic.name == "A"
    assert key.mode == "minor"


# ---------- transpose_to_target ----------

def test_transpose_d_major_to_c_major():
    # D major → C major: shift down 2 semitones.
    s = _build_one_bar_stream([("D4", 1), ("E4", 1), ("F#4", 1), ("G4", 1)])
    transposed, semitones = transpose_to_target(s, m21.key.Key("D", "major"))
    assert semitones == -2
    pitches = [n.nameWithOctave for n in transposed.recurse().notes]
    assert pitches[0] == "C4"
    assert pitches[1] == "D4"
    assert pitches[2] == "E4"
    assert pitches[3] == "F4"


def test_transpose_e_minor_to_a_minor():
    # E minor → A minor: shift up 5 semitones (or down 7 — music21 picks shortest).
    s = _build_one_bar_stream([("E4", 1), ("G4", 1)])
    transposed, semitones = transpose_to_target(s, m21.key.Key("E", "minor"))
    # Either +5 or -7 is acceptable musically; we lock the +5 choice
    # because Interval(E, A) gives a perfect 4th up (5 semitones).
    assert semitones == 5
    pitches = [n.nameWithOctave for n in transposed.recurse().notes]
    assert pitches[0] == "A4"
    assert pitches[1] == "C5"


def test_transpose_c_major_is_noop():
    s = _build_one_bar_stream([("C4", 1), ("E4", 1), ("G4", 1)])
    transposed, semitones = transpose_to_target(s, m21.key.Key("C", "major"))
    assert semitones == 0


# ---------- transpose_midi_back ----------

def test_transpose_midi_back_shifts_all_notes():
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
    pm.instruments.append(inst)

    transpose_midi_back(pm, semitones=-2)
    # We applied -2 → notes shift DOWN by 2 semitones.
    assert pm.instruments[0].notes[0].pitch == 58
    assert pm.instruments[0].notes[1].pitch == 62


def test_transpose_midi_back_noop_for_zero():
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    transpose_midi_back(pm, semitones=0)
    assert pm.instruments[0].notes[0].pitch == 60


def test_transpose_midi_back_handles_multiple_instruments():
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    for _ in range(3):
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        pm.instruments.append(inst)
    transpose_midi_back(pm, semitones=5)
    for inst in pm.instruments:
        assert inst.notes[0].pitch == 65
