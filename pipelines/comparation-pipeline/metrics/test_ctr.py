"""Unit tests for ChordToneRatio."""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21
import pretty_midi

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


def _score_with_chord(symbol: str) -> m21.stream.Score:
    """Score с одним ChordSymbol на offset 0."""
    score = m21.stream.Score()
    part = m21.stream.Part()
    cs = m21.harmony.ChordSymbol(symbol)
    cs.offset = 0.0
    part.insert(0.0, cs)
    score.append(part)
    return score


def test_ctr_all_chord_tones() -> None:
    """Все ноты — chord tones Cmaj7 (C, E, G, B) → CTR = 1.0."""
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = _pm_with_pitches([60, 64, 67, 71])  # C, E, G, B
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ChordToneRatio().compute(ctx) == 1.0


def test_ctr_no_chord_tones() -> None:
    """Все ноты вне Cmaj7 → CTR = 0.0."""
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = _pm_with_pitches([61, 63, 66, 68])  # C#, D#, F#, G# — все вне C/E/G/B
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ChordToneRatio().compute(ctx) == 0.0


def test_ctr_half_chord_tones() -> None:
    """2 из 4 нот — chord tones → CTR = 0.5."""
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = _pm_with_pitches([60, 61, 64, 63])  # C, C#, E, D# → C и E попадают
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ChordToneRatio().compute(ctx) == 0.5


def test_ctr_zero_notes_returns_none() -> None:
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert ChordToneRatio().compute(ctx) is None


def test_ctr_no_chord_context_returns_none() -> None:
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = _pm_with_pitches([60, 64, 67])
    ctx = SegmentContext(segment=pm, chord_context=None)
    assert ChordToneRatio().compute(ctx) is None


def test_ctr_empty_chord_context_returns_none() -> None:
    """Score без ChordSymbol → CTR = None."""
    from base import SegmentContext
    from ctr import ChordToneRatio
    pm = _pm_with_pitches([60, 64, 67])
    empty_score = m21.stream.Score()
    empty_score.append(m21.stream.Part())
    ctx = SegmentContext(segment=pm, chord_context=empty_score)
    assert ChordToneRatio().compute(ctx) is None


def test_ctr_name() -> None:
    from ctr import ChordToneRatio
    assert ChordToneRatio().name == "ctr"
