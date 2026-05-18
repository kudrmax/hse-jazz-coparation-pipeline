"""NoteDensity metric — count(notes) / bars."""
from __future__ import annotations

from base import Metric, SegmentContext


class NoteDensity(Metric):
    """Доля нот на такт.

    Returns: count(notes) / bars. На 0 нот возвращает 0.0 (не None) — пустой
    сегмент имеет легитимную плотность 0, в отличие от метрик где плотность
    'не определена' (например энтропия).
    """
    name = "note_density"

    def compute(self, ctx: SegmentContext) -> float | None:
        n_notes = sum(len(ins.notes) for ins in ctx.segment.instruments)
        return n_notes / ctx.bars
