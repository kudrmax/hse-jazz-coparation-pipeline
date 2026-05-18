"""ChordMatchPerTime metric — chord-tone ratio, взвешенный по длительности нот.

PER-NOTE vs PER-TIME: это per-time вариант ChordToneRatio (см. ctr.py).
Каждая нота вносит вклад, равный её длительности, поэтому долгие ноты
весят больше коротких. Соответствует определению из BebopNet §5.1
("percent of time within a measure where notes match pitches of the chord").
ChordToneRatio (ctr.py) считает per-note, что согласовано с MINGUS / CMT.
В ВКР держим обе метрики бок о бок именно ради этой методологической развилки.
"""
from __future__ import annotations

from base import Metric, SegmentContext
from ctr import _active_chord_for_offset, _extract_chord_offsets, _seconds_to_quarter


class ChordMatchPerTime(Metric):
    """Доля длительности нот сегмента, попадающая в chord tones активного аккорда.

    PER-NOTE vs PER-TIME (важно для ВКР):
        Это per-time вариант. Долгие ноты весят больше — формула BebopNet §5.1.
        Per-note версия — ChordToneRatio (ctr.py).
    """

    name = "chord_match_per_time"

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
            chord_pcs = {p.pitchClass for p in ch.pitches}
            dur = max(n.end - n.start, 0.0)
            total_dur += dur
            if (n.pitch % 12) in chord_pcs:
                hits_dur += dur
        if total_dur == 0.0:
            return None
        return hits_dur / total_dur
