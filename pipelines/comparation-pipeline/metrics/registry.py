"""Список всех метрик в порядке колонок per_segment_metrics.csv."""
from __future__ import annotations

from base import Metric
from chord_match_per_time import ChordMatchPerTime
from ctr import ChordToneRatio
from ctr_first_beat import CtrFirstBeat
from ngram_overlap import ThemeNgramOverlap
from note_density import NoteDensity
from pitch_entropy import PitchEntropy
from scale_match import ScaleMatch
from scale_match_per_time import ScaleMatchPerTime


def all_metrics() -> list[Metric]:
    """Возвращает список всех метрик в порядке колонок CSV."""
    return [
        ChordToneRatio(),
        ChordMatchPerTime(),
        CtrFirstBeat(),
        ScaleMatch(),
        ScaleMatchPerTime(),
        NoteDensity(),
        PitchEntropy(),
        ThemeNgramOverlap(n=3),
        ThemeNgramOverlap(n=4),
        ThemeNgramOverlap(n=5),
    ]
