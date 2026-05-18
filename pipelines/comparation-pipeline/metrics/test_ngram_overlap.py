"""Unit tests for ThemeNgramOverlap."""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21
import pretty_midi

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _pm_with_pitches(pitches: list[int]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(ins)
    return pm


def _score_with_pitches(pitches: list[int]) -> m21.stream.Score:
    """Score с одной part, ноты — четвертные подряд по тактам."""
    score = m21.stream.Score()
    part = m21.stream.Part()
    for i, p in enumerate(pitches):
        m = m21.stream.Measure(number=i + 1)
        m.append(m21.note.Note(p, quarterLength=4.0))
        part.append(m)
    score.append(part)
    return score


def test_ngram3_full_overlap() -> None:
    """Сегмент в точности повторяет 3 ноты из темы → overlap = 1.0 для n=3."""
    from base import SegmentContext
    from ngram_overlap import ThemeNgramOverlap
    pm_seg = _pm_with_pitches([60, 62, 64, 60])         # intervals [+2, +2, -4]
    cmp_score = _score_with_pitches([72, 74, 76, 72])   # intervals [+2, +2, -4] (тот же узор, выше октава)
    ctx = SegmentContext(segment=pm_seg, comparison_melody=cmp_score)
    assert ThemeNgramOverlap(n=3).compute(ctx) == 1.0


def test_ngram3_no_overlap() -> None:
    """Сегмент идёт вверх, тема — вниз; никаких 3-грамм не совпадает."""
    from base import SegmentContext
    from ngram_overlap import ThemeNgramOverlap
    pm_seg = _pm_with_pitches([60, 62, 64, 65])         # intervals [+2, +2, +1]
    cmp_score = _score_with_pitches([72, 70, 68, 67])   # intervals [-2, -2, -1]
    ctx = SegmentContext(segment=pm_seg, comparison_melody=cmp_score)
    assert ThemeNgramOverlap(n=3).compute(ctx) == 0.0


def test_ngram3_partial_overlap() -> None:
    """1 из 2 3-грамм совпадает → overlap = 0.5."""
    from base import SegmentContext
    from ngram_overlap import ThemeNgramOverlap
    pm_seg = _pm_with_pitches([60, 62, 64, 60, 62])     # intervals [+2, +2, -4, +2] → 3-grams: (+2,+2,-4), (+2,-4,+2)
    cmp_score = _score_with_pitches([72, 74, 76, 72])   # intervals [+2, +2, -4] → 3-grams: (+2,+2,-4)
    ctx = SegmentContext(segment=pm_seg, comparison_melody=cmp_score)
    assert ThemeNgramOverlap(n=3).compute(ctx) == 0.5


def test_ngram_too_few_notes_returns_none() -> None:
    """Сегмент имеет <n+1 нот → не получится n-грамм → None."""
    from base import SegmentContext
    from ngram_overlap import ThemeNgramOverlap
    pm_seg = _pm_with_pitches([60, 62])   # 2 ноты → 1 интервал → нет 3-грамм
    cmp_score = _score_with_pitches([60, 62, 64, 65])
    ctx = SegmentContext(segment=pm_seg, comparison_melody=cmp_score)
    assert ThemeNgramOverlap(n=3).compute(ctx) is None


def test_ngram_no_comparison_returns_none() -> None:
    from base import SegmentContext
    from ngram_overlap import ThemeNgramOverlap
    pm_seg = _pm_with_pitches([60, 62, 64, 65])
    ctx = SegmentContext(segment=pm_seg, comparison_melody=None)
    assert ThemeNgramOverlap(n=3).compute(ctx) is None


def test_ngram_name_includes_n() -> None:
    from ngram_overlap import ThemeNgramOverlap
    assert ThemeNgramOverlap(n=3).name == "ngram_3_overlap"
    assert ThemeNgramOverlap(n=5).name == "ngram_5_overlap"
