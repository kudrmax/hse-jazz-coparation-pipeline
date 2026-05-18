"""Тесты извлечения интервал-последовательностей из PrettyMIDI и m21.Score."""
from __future__ import annotations

import music21 as m21
import pretty_midi

from plagiarism.intervals import intervals_from_midi, intervals_from_score


def _make_pm(events: list[tuple[float, float, int]]) -> pretty_midi.PrettyMIDI:
    """events: list of (start_sec, end_sec, pitch_midi)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch in events:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def _make_score(pitches: list[str]) -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()
    for p in pitches:
        part.append(m21.note.Note(p, quarterLength=1.0))
    score.append(part)
    return score


def test_intervals_from_midi_basic_ascending():
    pm = _make_pm([(0.0, 0.5, 60), (0.5, 1.0, 64), (1.0, 1.5, 67)])
    assert intervals_from_midi(pm) == [4, 3]


def test_intervals_from_midi_descending_and_octave():
    pm = _make_pm([(0.0, 0.5, 72), (0.5, 1.0, 60), (1.0, 1.5, 79)])
    assert intervals_from_midi(pm) == [-12, 19]


def test_intervals_from_midi_sorts_by_start():
    """Внеочередные ноты должны сортироваться по start."""
    pm = _make_pm([(1.0, 1.5, 67), (0.0, 0.5, 60), (0.5, 1.0, 64)])
    assert intervals_from_midi(pm) == [4, 3]


def test_intervals_from_midi_empty():
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    assert intervals_from_midi(pm) == []


def test_intervals_from_midi_single_note():
    pm = _make_pm([(0.0, 0.5, 60)])
    assert intervals_from_midi(pm) == []


def test_intervals_from_score_basic():
    score = _make_score(["C4", "E4", "G4"])
    assert intervals_from_score(score) == [4, 3]


def test_intervals_from_score_ignores_chord_symbol():
    score = m21.stream.Score()
    part = m21.stream.Part()
    part.append(m21.note.Note("C4", quarterLength=1.0))
    part.append(m21.harmony.ChordSymbol("Cmaj7", quarterLength=0.0))
    part.append(m21.note.Note("E4", quarterLength=1.0))
    score.append(part)
    assert intervals_from_score(score) == [4]


def test_intervals_from_score_single_note():
    score = _make_score(["C4"])
    assert intervals_from_score(score) == []


def test_intervals_from_score_empty():
    score = m21.stream.Score()
    assert intervals_from_score(score) == []
