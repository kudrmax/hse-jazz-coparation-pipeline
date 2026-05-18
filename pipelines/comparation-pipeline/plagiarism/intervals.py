"""Извлечение последовательности интервалов (pitch[i+1]-pitch[i]) из
pretty_midi.PrettyMIDI и music21.stream.Score.

Transposition-invariant: интервалы не зависят от абсолютной тональности,
что нужно и для n-gram overlap, и для LCS.
"""
from __future__ import annotations

import music21 as m21
import pretty_midi


def intervals_from_midi(pm: pretty_midi.PrettyMIDI) -> list[int]:
    """Все ноты со всех instruments → сортировка по (start, pitch) → интервалы.

    Возвращает [] если <2 нот.
    """
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    if len(notes) < 2:
        return []
    pitches = [n.pitch for n in notes]
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def intervals_from_score(score: m21.stream.Score) -> list[int]:
    """recurse().getElementsByClass(m21.note.Note) → pitch.midi → интервалы.

    ChordSymbol и Chord игнорируются (для нашей monophonic melody их быть
    не должно, но защита).

    Возвращает [] если <2 нот.
    """
    pitches: list[int] = [
        el.pitch.midi
        for el in score.recurse().getElementsByClass(m21.note.Note)
    ]
    if len(pitches) < 2:
        return []
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
