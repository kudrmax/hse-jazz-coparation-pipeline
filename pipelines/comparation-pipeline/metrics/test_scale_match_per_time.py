"""Unit tests for ScaleMatchPerTime."""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21
import pretty_midi
import pytest

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _pm_with_pitches(pitches: list[int], note_dur: float = 0.5) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + note_dur))
        t += note_dur
    pm.instruments.append(ins)
    return pm


def _pm_with_pitch_durs(specs: list[tuple[int, float]]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for pitch, dur in specs:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=t, end=t + dur))
        t += dur
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


def test_scale_match_per_time_all_in_scale() -> None:
    """Все ноты — в C major scale (Cmaj7) → 1.0."""
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = _pm_with_pitches([60, 62, 64, 65, 67, 69, 71])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatchPerTime().compute(ctx) == 1.0


def test_scale_match_per_time_none_in_scale() -> None:
    """Все вне C major scale → 0.0. Пентатоника F# (чёрные клавиши)."""
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = _pm_with_pitches([61, 63, 66, 68, 70])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatchPerTime().compute(ctx) == 0.0


def test_scale_match_per_time_long_in_scale_weighs_more() -> None:
    """In-scale (2.0s) + out-of-scale (0.5s) на Cmaj7 → 2.0/2.5 = 0.8."""
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = _pm_with_pitch_durs([(60, 2.0), (61, 0.5)])  # C in, C# out
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    result = ScaleMatchPerTime().compute(ctx)
    assert result is not None
    assert result == pytest.approx(0.8)


def test_scale_match_per_time_long_out_of_scale_weighs_more() -> None:
    """In-scale (0.5s) + out-of-scale (2.0s) на Cmaj7 → 0.5/2.5 = 0.2."""
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = _pm_with_pitch_durs([(60, 0.5), (61, 2.0)])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    result = ScaleMatchPerTime().compute(ctx)
    assert result is not None
    assert result == pytest.approx(0.2)


def test_scale_match_per_time_zero_notes_returns_none() -> None:
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ScaleMatchPerTime().compute(ctx) is None


def test_scale_match_per_time_no_chord_context_returns_none() -> None:
    from base import SegmentContext
    from scale_match_per_time import ScaleMatchPerTime
    pm = _pm_with_pitches([60, 64, 67])
    ctx = SegmentContext(segment=pm, chord_context=None)
    assert ScaleMatchPerTime().compute(ctx) is None


def test_scale_match_per_time_name() -> None:
    from scale_match_per_time import ScaleMatchPerTime
    assert ScaleMatchPerTime().name == "scale_match_per_time"
