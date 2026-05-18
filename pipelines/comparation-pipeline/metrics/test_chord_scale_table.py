"""Unit tests for chord_scale_table — chord type → scale class lookup."""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def test_cmaj7_to_c_major_scale() -> None:
    from chord_scale_table import scale_pcs_for_chord
    cs = m21.harmony.ChordSymbol("Cmaj7")
    pcs = scale_pcs_for_chord(cs)
    # C major: C D E F G A B → pc 0,2,4,5,7,9,11
    assert pcs == {0, 2, 4, 5, 7, 9, 11}


def test_dm7_to_d_dorian_scale() -> None:
    from chord_scale_table import scale_pcs_for_chord
    cs = m21.harmony.ChordSymbol("Dm7")
    pcs = scale_pcs_for_chord(cs)
    # D dorian: D E F G A B C → pc 2,4,5,7,9,11,0
    assert pcs == {0, 2, 4, 5, 7, 9, 11}


def test_g7_to_g_mixolydian_scale() -> None:
    from chord_scale_table import scale_pcs_for_chord
    cs = m21.harmony.ChordSymbol("G7")
    pcs = scale_pcs_for_chord(cs)
    # G mixolydian: G A B C D E F → pc 7,9,11,0,2,4,5
    assert pcs == {0, 2, 4, 5, 7, 9, 11}


def test_unknown_chord_falls_back_to_major() -> None:
    """Незнакомый тип → fallback на major scale of root."""
    from chord_scale_table import scale_pcs_for_chord
    cs = m21.harmony.ChordSymbol("C")  # триада, тип "major"
    pcs = scale_pcs_for_chord(cs)
    assert pcs == {0, 2, 4, 5, 7, 9, 11}  # C major


def test_minor_triad_uses_dorian() -> None:
    from chord_scale_table import scale_pcs_for_chord
    cs = m21.harmony.ChordSymbol("Cm")
    pcs = scale_pcs_for_chord(cs)
    # C dorian: C D Eb F G A Bb → pc 0,2,3,5,7,9,10
    assert pcs == {0, 2, 3, 5, 7, 9, 10}
