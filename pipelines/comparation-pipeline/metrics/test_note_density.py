"""Unit tests for NoteDensity."""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _make_pm(pitches: list[int], note_dur: float = 0.5) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + note_dur))
        t += note_dur
    pm.instruments.append(ins)
    return pm


def test_note_density_simple() -> None:
    from base import SegmentContext
    from note_density import NoteDensity
    pm = _make_pm([60, 62, 64, 65, 67, 69, 71, 72])  # 8 нот
    ctx = SegmentContext(segment=pm, bars=8)
    assert NoteDensity().compute(ctx) == 1.0  # 8 нот / 8 тактов


def test_note_density_two_notes_in_eight_bars() -> None:
    from base import SegmentContext
    from note_density import NoteDensity
    pm = _make_pm([60, 62])
    ctx = SegmentContext(segment=pm, bars=8)
    assert NoteDensity().compute(ctx) == 0.25


def test_note_density_zero_notes_returns_zero() -> None:
    """Спека: 0 нот → 0.0 (не None) для NoteDensity."""
    from base import SegmentContext
    from note_density import NoteDensity
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))  # пусто
    ctx = SegmentContext(segment=pm, bars=8)
    assert NoteDensity().compute(ctx) == 0.0


def test_note_density_name() -> None:
    from note_density import NoteDensity
    assert NoteDensity().name == "note_density"
