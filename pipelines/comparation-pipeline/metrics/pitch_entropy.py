"""PitchEntropy metric — Shannon entropy распределения pitch classes."""
from __future__ import annotations

import math
from collections import Counter

from base import Metric, SegmentContext


class PitchEntropy(Metric):
    """Энтропия Shannon по распределению pitch class'ов (mod 12).

    Returns: H = -Σ p_i * log2(p_i), где p_i — частота i-го pitch class.
        Диапазон: 0 (все ноты — один pc) до log2(12) ≈ 3.585 (равномерно).
        На 0 нот → None.
    """
    name = "pitch_entropy"

    def compute(self, ctx: SegmentContext) -> float | None:
        pitches = [n.pitch for ins in ctx.segment.instruments for n in ins.notes]
        if not pitches:
            return None
        counts = Counter(p % 12 for p in pitches)
        total = sum(counts.values())
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * math.log2(p)
        return h
