"""ScaleMatchPerTime metric — scale-match ratio, взвешенный по длительности нот.

PER-NOTE vs PER-TIME: это per-time вариант ScaleMatch (см. scale_match.py).
Каждая нота вносит вклад, равный её длительности, поэтому долгие ноты
весят больше коротких. Соответствует определению из BebopNet §5.1
("percent of time within a measure where notes match pitches of the scale").
ScaleMatch (scale_match.py) считает per-note, что согласовано с MINGUS / CMT.
В ВКР держим обе метрики бок о бок именно ради этой методологической развилки.
"""
from __future__ import annotations

from base import Metric, SegmentContext
from chord_scale_table import scale_pcs_for_chord
from ctr import _active_chord_for_offset, _extract_chord_offsets, _seconds_to_quarter


class ScaleMatchPerTime(Metric):
    """Доля длительности нот сегмента, попадающая в гамму активного аккорда.

    PER-NOTE vs PER-TIME (важно для ВКР):
        Это per-time вариант. Долгие ноты весят больше — формула BebopNet §5.1.
        Per-note версия — ScaleMatch (scale_match.py).
    """

    name = "scale_match_per_time"

    def compute(self, ctx: SegmentContext) -> float | None:
        notes = [n for ins in ctx.segment.instruments for n in ins.notes]
        if not notes:
            return None
        if ctx.chord_context is None:
            return None
        chords = _extract_chord_offsets(ctx.chord_context)
        if not chords:
            return None

        hits_dur = 0.0
        total_dur = 0.0
        for n in notes:
            offset_q = _seconds_to_quarter(n.start, ctx.segment)
            ch = _active_chord_for_offset(chords, offset_q)
            if ch is None:
                continue
            scale_pcs = scale_pcs_for_chord(ch)
            if not scale_pcs:
                continue
            dur = max(n.end - n.start, 0.0)
            total_dur += dur
            if (n.pitch % 12) in scale_pcs:
                hits_dur += dur
        if total_dur == 0.0:
            return None
        return hits_dur / total_dur
