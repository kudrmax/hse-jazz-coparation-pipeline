"""ChordToneRatio metric — доля нот, попадающих в chord tones активного аккорда.

PER-NOTE vs PER-TIME: считаем per-note (как MINGUS, CMT). BebopNet считает
per-time, что даёт больший вес длинным нотам. Мы намеренно используем per-note
для согласованности с двумя из трёх ссылочных моделей. В ВКР это нужно
явно отметить — иначе рецензент спросит почему наши цифры BebopNet
расходятся с таблицей в их статье.
"""
from __future__ import annotations

import bisect

import music21 as m21

from base import Metric, SegmentContext


def _extract_chord_offsets(chord_context: m21.stream.Score) -> list[tuple[float, m21.harmony.ChordSymbol]]:
    """Возвращает [(offset_quarter, ChordSymbol), ...] отсортированный по offset."""
    chords: list[tuple[float, m21.harmony.ChordSymbol]] = []
    for cs in chord_context.recurse().getElementsByClass(m21.harmony.ChordSymbol):
        chords.append((float(cs.offset), cs))
    chords.sort(key=lambda x: x[0])
    return chords


def _seconds_to_quarter(seconds: float, pm: "pretty_midi.PrettyMIDI") -> float:  # type: ignore[name-defined]
    """Конвертирует секунды в quarter offset через темп pretty_midi."""
    tempo_times, tempo_values = pm.get_tempo_changes()
    if tempo_values.size > 0:
        bpm = float(tempo_values[0])
    else:
        bpm = 120.0
    return seconds * (bpm / 60.0)


def _active_chord_for_offset(
    chords: list[tuple[float, m21.harmony.ChordSymbol]],
    offset_q: float,
) -> m21.harmony.ChordSymbol | None:
    """Последний аккорд с offset ≤ offset_q. Bisect right + 1 шаг назад."""
    if not chords:
        return None
    offsets_only = [o for o, _ in chords]
    idx = bisect.bisect_right(offsets_only, offset_q) - 1
    if idx < 0:
        return None
    return chords[idx][1]


class ChordToneRatio(Metric):
    """Доля нот сегмента, попадающих в chord tones активного аккорда.

    PER-NOTE vs PER-TIME (важно для ВКР):
        Считаем per-note (как MINGUS, CMT). BebopNet использует per-time,
        что даёт больший вес длинным нотам. Расхождение намеренное —
        нужно явно зафиксировать в работе.

    Source: Cooke et al. (2002) chord-scale system; BebopNet/MINGUS papers.
    """

    name = "ctr"

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
            chord_pcs = {p.pitchClass for p in ch.pitches}
            total += 1
            if (n.pitch % 12) in chord_pcs:
                hits += 1
        if total == 0:
            return None
        return hits / total
