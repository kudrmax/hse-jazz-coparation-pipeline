"""CtrFirstBeat metric — chord-tone ratio только по нотам на первой доле такта.

PER-NOTE vs PER-TIME: per-note CTR (как ChordToneRatio), но с предварительным
фильтром на ноты, попадающие на downbeat'ы (первую долю каждого такта).
BebopNet версия (per-time, без downbeat-фильтра) — ChordMatchPerTime.
В ВКР держим эти метрики раздельно ради методологической прозрачности.

Источник: CMT (Choi et al. 2021) §V.A.2 — "chord tone ratio of the first beat
of each bar" — авторы используют именно эту разновидность как ключевой
показатель chord match.
"""
from __future__ import annotations

import bisect

from base import Metric, SegmentContext
from ctr import _active_chord_for_offset, _extract_chord_offsets, _seconds_to_quarter


class CtrFirstBeat(Metric):
    """Per-note CTR, посчитанный только по нотам, стартующим на downbeat'е.

    Tolerance "нота на downbeat'е": 1/16 четверти, конвертированная в
    секунды через текущий темп pretty_midi.

    PER-NOTE vs PER-TIME (важно для ВКР):
        Per-note (как ChordToneRatio). Per-time эквивалента у CMT нет;
        BebopNet per-time — отдельная метрика ChordMatchPerTime.
    """

    name = "ctr_first_beat"

    def compute(self, ctx: SegmentContext) -> float | None:
        notes = [n for ins in ctx.segment.instruments for n in ins.notes]
        if not notes:
            return None
        if ctx.chord_context is None:
            return None
        chords = _extract_chord_offsets(ctx.chord_context)
        if not chords:
            return None

        downbeats_arr = ctx.segment.get_downbeats()
        downbeats = sorted(float(d) for d in downbeats_arr)
        if not downbeats:
            return None

        tempo_times, tempo_values = ctx.segment.get_tempo_changes()
        bpm = float(tempo_values[0]) if tempo_values.size > 0 else 120.0
        tolerance_sec = (1.0 / 16.0) * 60.0 / bpm

        on_downbeat: list = []
        for n in notes:
            i = bisect.bisect_left(downbeats, n.start)
            candidates = []
            if i > 0:
                candidates.append(downbeats[i - 1])
            if i < len(downbeats):
                candidates.append(downbeats[i])
            if not candidates:
                continue
            nearest_diff = min(abs(n.start - c) for c in candidates)
            if nearest_diff <= tolerance_sec:
                on_downbeat.append(n)

        if not on_downbeat:
            return None

        hits = 0
        total = 0
        for n in on_downbeat:
            offset_q = _seconds_to_quarter(n.start, ctx.segment)
            ch = _active_chord_for_offset(chords, offset_q)
            if ch is None:
                continue
            chord_pcs = {p.pitchClass for p in ch.pitches}
            total += 1
            if (n.pitch % 12) in chord_pcs:
                hits += 1
        if total == 0:
            return None
        return hits / total
