"""Unit tests for PitchEntropy."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pretty_midi

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _make_pm(pitches: list[int]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(ins)
    return pm


def test_pitch_entropy_all_same_pc_zero() -> None:
    """Все ноты — один pitch class C → энтропия 0."""
    from base import SegmentContext
    from pitch_entropy import PitchEntropy
    pm = _make_pm([60, 60, 60, 72, 72])  # все C, разные октавы → pc одинаковые
    ctx = SegmentContext(segment=pm)
    assert PitchEntropy().compute(ctx) == 0.0


def test_pitch_entropy_uniform_two_classes() -> None:
    """Два pitch class'а 50/50 → энтропия 1 бит."""
    from base import SegmentContext
    from pitch_entropy import PitchEntropy
    pm = _make_pm([60, 62, 60, 62])  # C, D, C, D
    ctx = SegmentContext(segment=pm)
    assert abs(PitchEntropy().compute(ctx) - 1.0) < 1e-9


def test_pitch_entropy_uniform_all_12() -> None:
    """Все 12 pitch classes по одной → log2(12) ≈ 3.585."""
    from base import SegmentContext
    from pitch_entropy import PitchEntropy
    pm = _make_pm(list(range(60, 72)))
    ctx = SegmentContext(segment=pm)
    expected = math.log2(12)
    assert abs(PitchEntropy().compute(ctx) - expected) < 1e-9


def test_pitch_entropy_zero_notes_returns_none() -> None:
    from base import SegmentContext
    from pitch_entropy import PitchEntropy
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    ctx = SegmentContext(segment=pm)
    assert PitchEntropy().compute(ctx) is None


def test_pitch_entropy_name() -> None:
    from pitch_entropy import PitchEntropy
    assert PitchEntropy().name == "pitch_entropy"
