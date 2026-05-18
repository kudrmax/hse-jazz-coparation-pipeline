"""ThemeNgramOverlap metric — n-граммы интервалов сегмента ∩ comparison_melody.

Transposition-invariant: считаем последовательности интервалов между соседними
нотами (полутоны), а не абсолютные pitch'и. Так модель в другой тональности
с теми же мелодическими паттернами всё равно даёт высокий overlap.
"""
from __future__ import annotations

import music21 as m21
import pretty_midi

from base import Metric, SegmentContext


def _pitch_sequence_pm(pm: pretty_midi.PrettyMIDI) -> list[int]:
    """Извлекает pitch'и в порядке start time из всех инструментов pm."""
    notes = [n for ins in pm.instruments for n in ins.notes]
    notes.sort(key=lambda n: n.start)
    return [n.pitch for n in notes]


def _pitch_sequence_score(score: m21.stream.Score) -> list[int]:
    """Извлекает pitch'и в порядке offset из m21 Score (только Note, не Chord/ChordSymbol)."""
    seq: list[int] = []
    for el in score.recurse().getElementsByClass(m21.note.Note):
        seq.append(el.pitch.midi)
    return seq


def _intervals(pitches: list[int]) -> list[int]:
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def _ngrams(seq: list[int], n: int) -> set[tuple[int, ...]]:
    if len(seq) < n:
        return set()
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


class ThemeNgramOverlap(Metric):
    """Доля n-грамм интервалов сегмента, встречающихся в comparison_melody.

    Args:
        n: длина n-граммы (3, 4 или 5 в нашем pipeline).

    Returns:
        |seg_ngrams ∩ cmp_ngrams| / |seg_ngrams|. На <n+1 нотах в сегменте
        или отсутствии comparison_melody → None.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.name = f"ngram_{n}_overlap"

    def compute(self, ctx: SegmentContext) -> float | None:
        if ctx.comparison_melody is None:
            return None
        seg_pitches = _pitch_sequence_pm(ctx.segment)
        cmp_pitches = _pitch_sequence_score(ctx.comparison_melody)
        seg_intervals = _intervals(seg_pitches)
        cmp_intervals = _intervals(cmp_pitches)
        seg_ng = _ngrams(seg_intervals, self.n)
        if not seg_ng:
            return None
        cmp_ng = _ngrams(cmp_intervals, self.n)
        if not cmp_ng:
            return 0.0
        return len(seg_ng & cmp_ng) / len(seg_ng)
