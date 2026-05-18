"""Unit tests for ScaleMatch."""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21
import pretty_midi

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _pm_with_pitches(pitches: list[int]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(ins)
    return pm


def _score_with_chord(symbol: str) -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()
    cs = m21.harmony.ChordSymbol(symbol)
    cs.offset = 0.0
    part.insert(0.0, cs)
    score.append(part)
    return score


def test_scale_match_all_in_c_major_for_cmaj7() -> None:
    """Все ноты — в C major scale → SM = 1.0."""
    from base import SegmentContext
    from scale_match import ScaleMatch
    pm = _pm_with_pitches([60, 62, 64, 65, 67, 69, 71])  # C D E F G A B
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatch().compute(ctx) == 1.0


def test_scale_match_dm7_dorian() -> None:
    """Все ноты в D dorian → SM = 1.0 для Dm7."""
    from base import SegmentContext
    from scale_match import ScaleMatch
    pm = _pm_with_pitches([62, 64, 65, 67, 69, 71, 72])  # D E F G A B C
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Dm7"))
    assert ScaleMatch().compute(ctx) == 1.0


def test_scale_match_chromatic_partial() -> None:
    """Ноты вне C major scale (например F#) не попадают в SM."""
    from base import SegmentContext
    from scale_match import ScaleMatch
    pm = _pm_with_pitches([60, 61, 64, 66])  # C, C#, E, F# → 2 из 4 в C-major
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatch().compute(ctx) == 0.5


def test_scale_match_zero_notes_returns_none() -> None:
    from base import SegmentContext
    from scale_match import ScaleMatch
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatch().compute(ctx) is None


def test_scale_match_no_chord_context_returns_none() -> None:
    from base import SegmentContext
    from scale_match import ScaleMatch
    pm = _pm_with_pitches([60, 64])
    ctx = SegmentContext(segment=pm, chord_context=None)
    assert ScaleMatch().compute(ctx) is None


def test_scale_match_name() -> None:
    from scale_match import ScaleMatch
    assert ScaleMatch().name == "scale_match"
