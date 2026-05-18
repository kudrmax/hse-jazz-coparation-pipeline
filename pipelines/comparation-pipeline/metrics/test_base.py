"""Unit tests for base.py — Metric ABC + SegmentContext dataclass."""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def test_segment_context_minimal_construction() -> None:
    """SegmentContext конструируется с минимальным набором — только segment+bars."""
    from base import SegmentContext
    pm = pretty_midi.PrettyMIDI()
    ctx = SegmentContext(segment=pm)
    assert ctx.segment is pm
    assert ctx.chord_context is None
    assert ctx.comparison_melody is None
    assert ctx.bars == 8


def test_segment_context_frozen() -> None:
    """Dataclass frozen — нельзя мутировать."""
    from base import SegmentContext
    pm = pretty_midi.PrettyMIDI()
    ctx = SegmentContext(segment=pm)
    with pytest.raises(Exception):  # FrozenInstanceError
        ctx.bars = 16  # type: ignore


def test_metric_abc_cannot_instantiate() -> None:
    """Metric — abstract, не должен инстанциироваться напрямую."""
    from base import Metric
    with pytest.raises(TypeError):
        Metric()  # type: ignore


def test_metric_subclass_implements_compute() -> None:
    """Конкретный наследник имеет name и compute()."""
    from base import Metric, SegmentContext

    class DummyMetric(Metric):
        name = "dummy"
        def compute(self, ctx: SegmentContext) -> float | None:
            return 42.0

    pm = pretty_midi.PrettyMIDI()
    ctx = SegmentContext(segment=pm)
    m = DummyMetric()
    assert m.name == "dummy"
    assert m.compute(ctx) == 42.0
