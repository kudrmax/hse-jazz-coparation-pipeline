"""Helpers для построения synthetic m21 Measure'ов в bar_rhythm_jsd тестах."""
from __future__ import annotations

import music21 as m21


def make_measure(events: list[tuple[float, float, str | None]]) -> m21.stream.Measure:
    """Построить 4/4 Measure из списка (offset_in_q, length_in_q, pitch | None).

    None pitch → Rest. Offsets могут быть Fraction-friendly числа (0.25, 0.5,
    1/3 и т.д.). Если события не покрывают все 4 quarter — рест заполняется
    автоматически через m21 при экспорте, в наших тестах достаточно положить
    Rest вручную где нужно.
    """
    m = m21.stream.Measure(number=1)
    for offset, length, pitch in events:
        if pitch is None:
            el: m21.base.Music21Object = m21.note.Rest(quarterLength=length)
        else:
            el = m21.note.Note(pitch, quarterLength=length)
        m.insert(float(offset), el)
    return m
