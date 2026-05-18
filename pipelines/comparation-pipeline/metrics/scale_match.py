"""ScaleMatch metric — доля нот в гамме активного аккорда.

PER-NOTE vs PER-TIME: считаем per-note (как MINGUS, CMT). BebopNet использует
per-time. Расхождение намеренное (см. CTR docstring).

Source: Cooke et al. (2002) chord-scale system; BebopNet/MINGUS papers.
"""
from __future__ import annotations

from base import Metric, SegmentContext
from chord_scale_table import scale_pcs_for_chord
from ctr import _active_chord_for_offset, _extract_chord_offsets, _seconds_to_quarter


class ScaleMatch(Metric):
    """Доля нот сегмента, попадающих в гамму активного аккорда.

    PER-NOTE vs PER-TIME (важно для ВКР):
        Считаем per-note (как MINGUS, CMT). BebopNet использует per-time,
        что даёт больший вес длинным нотам. Расхождение намеренное —
        нужно явно зафиксировать в работе.
    """

    name = "scale_match"

    def compute(self, ctx: SegmentContext) -> float | None:
        notes = [n for ins in ctx.segment.instruments for n in ins.notes]
        if not notes:
            return None
        if ctx.chord_context is None:
            return None
        chords = _extract_chord_offsets(ctx.chord_context)
        if not chords:
            return None

        hits = 0
        total = 0
        for n in notes:
            offset_q = _seconds_to_quarter(n.start, ctx.segment)
            ch = _active_chord_for_offset(chords, offset_q)
            if ch is None:
                continue
            scale_pcs = scale_pcs_for_chord(ch)
            if not scale_pcs:
                continue
            total += 1
            if (n.pitch % 12) in scale_pcs:
                hits += 1
        if total == 0:
            return None
        return hits / total
