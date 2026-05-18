"""Generic ABC + dataclass для всех метрик comparation-pipeline.

Метрики НЕ знают про структуру outputs/<theme>/<model>/. Они принимают
SegmentContext с минимальными данными для расчёта; caller (compute_metrics.py)
сам решает что подставить.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import music21 as m21
import pretty_midi


@dataclass(frozen=True)
class SegmentContext:
    """Generic вход для метрики.

    segment: сама мелодия сегмента (всегда требуется).
    chord_context: источник аккордов (нужен CTR/ScaleMatch).
    comparison_melody: с чем сравнивать (нужен NgramOverlap/ContourSimilarity).
    bars: длина сегмента в тактах (по умолчанию 8).
    """
    segment: pretty_midi.PrettyMIDI
    chord_context: m21.stream.Score | None = None
    comparison_melody: m21.stream.Score | None = None
    bars: int = 8


class Metric(ABC):
    """Все метрики наследуют отсюда. Каждая возвращает float или None
    (None — если метрика неопределена для этого сегмента: 0 нот, 1 нота, и т.п.).
    """
    name: str

    @abstractmethod
    def compute(self, ctx: SegmentContext) -> float | None:
        ...
