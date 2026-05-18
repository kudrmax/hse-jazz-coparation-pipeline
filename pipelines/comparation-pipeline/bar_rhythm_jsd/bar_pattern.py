"""Извлечение 16-character ритмического паттерна из одного m21 Measure'а.

Алфавит: 'O' onset, 'H' hold, 'R' rest. 16 slots (16-х долей на 4/4 такт).
Triplet'ы snap'аются к ближайшему 16-му (известное ограничение, см. spec).
"""
from __future__ import annotations

import music21 as m21

_SLOTS_PER_BAR = 16
_SLOTS_PER_QUARTER = 4  # 16 / 4


def extract_bar_pattern(measure: m21.stream.Measure) -> str:
    """Преобразовать Measure в 16-символьную строку из {O, H, R}."""
    slots: list[str] = ["R"] * _SLOTS_PER_BAR
    for note in measure.flatten().notes:
        if isinstance(note, m21.harmony.ChordSymbol):
            continue
        onset_slot = round(float(note.offset) * _SLOTS_PER_QUARTER)
        if onset_slot < 0 or onset_slot >= _SLOTS_PER_BAR:
            continue
        slot_len = max(1, round(float(note.duration.quarterLength) * _SLOTS_PER_QUARTER))
        end_slot = min(_SLOTS_PER_BAR, onset_slot + slot_len)
        slots[onset_slot] = "O"
        for k in range(onset_slot + 1, end_slot):
            if slots[k] == "R":
                slots[k] = "H"
    return "".join(slots)
