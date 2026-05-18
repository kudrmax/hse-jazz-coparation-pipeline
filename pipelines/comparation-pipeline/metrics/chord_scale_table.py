"""Hardcoded chord-type → m21 Scale class маппинг для ScaleMatch.

Источник теории: Cooke, Horn, Cross (2002) "The Cambridge Companion to Jazz" —
chord-scale system. Ссылается также BebopNet в статье. Точная таблица на
~10 типов покрывает 90%+ jazz lead sheet'ов; редкие случаи (alt, sus2, etc)
fallback на root major.
"""
from __future__ import annotations

import music21 as m21


# music21 chordKind → ScaleClass.
# Берём dorian для минорных (jazz default), mixolydian для доминант.
_KIND_TO_SCALE: dict[str, type] = {
    # major family
    "major":               m21.scale.MajorScale,
    "major-seventh":       m21.scale.MajorScale,
    "major-sixth":         m21.scale.MajorScale,
    "major-ninth":         m21.scale.MajorScale,
    # minor family → dorian (jazz default)
    "minor":               m21.scale.DorianScale,
    "minor-seventh":       m21.scale.DorianScale,
    "minor-ninth":         m21.scale.DorianScale,
    "minor-sixth":         m21.scale.DorianScale,
    # dominant → mixolydian
    "dominant":            m21.scale.MixolydianScale,
    "dominant-seventh":    m21.scale.MixolydianScale,
    "dominant-ninth":      m21.scale.MixolydianScale,
    "dominant-13th":       m21.scale.MixolydianScale,
    # half-diminished → locrian
    "half-diminished":     m21.scale.LocrianScale,
    # diminished → octatonic (whole-half)
    "diminished":          m21.scale.OctatonicScale,
    "diminished-seventh":  m21.scale.OctatonicScale,
    # augmented → whole-tone
    "augmented":           m21.scale.WholeToneScale,
    # sus → mixolydian (sus dominant) или major (sus2)
    "suspended-fourth":    m21.scale.MixolydianScale,
    "suspended-second":    m21.scale.MajorScale,
}

_FALLBACK_SCALE = m21.scale.MajorScale


def scale_pcs_for_chord(chord: m21.harmony.ChordSymbol) -> set[int]:
    """Возвращает set pitch class'ов гаммы, ассоциированной с аккордом.

    Args:
        chord: m21.harmony.ChordSymbol — например ChordSymbol('Cmaj7').

    Returns:
        Set int (0..11), pitch class'ы нот гаммы.
    """
    kind = chord.chordKind
    scale_cls = _KIND_TO_SCALE.get(kind, _FALLBACK_SCALE)
    root = chord.root()
    if root is None:
        return set()
    scale = scale_cls(root)
    return {p.pitchClass for p in scale.getPitches()}
