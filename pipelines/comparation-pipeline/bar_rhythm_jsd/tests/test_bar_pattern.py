"""Тесты extract_bar_pattern."""
from __future__ import annotations

from bar_rhythm_jsd.bar_pattern import extract_bar_pattern
from bar_rhythm_jsd.tests._helpers import make_measure


def test_four_quarter_notes_pattern():
    # 4 четверти на 0, 1, 2, 3
    m = make_measure([
        (0.0, 1.0, "C4"),
        (1.0, 1.0, "E4"),
        (2.0, 1.0, "G4"),
        (3.0, 1.0, "C5"),
    ])
    pattern = extract_bar_pattern(m)
    assert pattern == "OHHHOHHHOHHHOHHH"


def test_full_rest_bar():
    m = make_measure([(0.0, 4.0, None)])
    pattern = extract_bar_pattern(m)
    assert pattern == "RRRRRRRRRRRRRRRR"


def test_sixteenth_notes_all_onsets():
    # 16 шестнадцатых на offsets 0, 0.25, 0.5, ..., 3.75
    events = [(i * 0.25, 0.25, "C4") for i in range(16)]
    m = make_measure(events)
    pattern = extract_bar_pattern(m)
    assert pattern == "O" * 16


def test_eighth_notes_pattern():
    # 8 восьмушек на 0, 0.5, 1.0, ..., 3.5
    events = [(i * 0.5, 0.5, "C4") for i in range(8)]
    m = make_measure(events)
    pattern = extract_bar_pattern(m)
    # каждый onset slot 0,2,4,...,14 → O; nечётные slot'ы 1,3,5,... → H
    assert pattern == "OHOHOHOHOHOHOHOH"


def test_mixed_quarter_and_eighth():
    # quarter on 0, two eighths on 1 and 1.5, quarter on 2, quarter from 3
    m = make_measure([
        (0.0, 1.0, "C4"),
        (1.0, 0.5, "D4"),
        (1.5, 0.5, "E4"),
        (2.0, 1.0, "F4"),
        (3.0, 1.0, "G4"),
    ])
    pattern = extract_bar_pattern(m)
    # slots 0..3 quarter C → OHHH; slot 4 D onset → O; 5 D hold → H;
    # slot 6 E onset → O; 7 E hold → H
    # 8 F onset → O, 9..11 hold → HHH; 12 G onset → O, 13..15 hold → HHH
    assert pattern == "OHHHOHOHOHHHOHHH"


def test_eighth_triplet_snaps_to_16th_grid():
    # три триольные восьмые на 0, 1/3, 2/3
    m = make_measure([
        (0.0, 1.0 / 3, "C4"),
        (1.0 / 3, 1.0 / 3, "D4"),
        (2.0 / 3, 1.0 / 3, "E4"),
        # остальные 3 quarters — rest
        (1.0, 3.0, None),
    ])
    pattern = extract_bar_pattern(m)
    # round(0 * 4) = 0; round(1/3 * 4) = round(1.333) = 1;
    # round(2/3 * 4) = round(2.667) = 3.
    # Onsets на slots 0, 1, 3.
    assert len(pattern) == 16
    assert pattern.count("O") == 3
    assert pattern[0] == "O"
    assert pattern[1] == "O"
    assert pattern[3] == "O"
