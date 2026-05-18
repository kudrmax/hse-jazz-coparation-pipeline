"""Unit tests for CtrFirstBeat.

Замечание про downbeats: при `PrettyMIDI(initial_tempo=120.0)` без явного
TimeSignature pretty_midi возвращает downbeats на 0.0, 2.0, 4.0, ... s
(4 четверти/такт * 0.5 s/четверть = 2 s/такт). Tolerance в коде:
1/16 четверти = 0.0625 q * (60/120) = 0.03125 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import music21 as m21
import pretty_midi

METRICS_DIR = Path(__file__).resolve().parent
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))


def _pm_with_notes(specs: list[tuple[int, float, float]]) -> pretty_midi.PrettyMIDI:
    """specs: [(pitch, start_seconds, dur_seconds), ...]."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=0)
    for pitch, start, dur in specs:
        ins.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + dur))
    pm.instruments.append(ins)
    return pm


def _score_with_chord(symbol: str) -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()
    cs = m21.harmony.ChordSymbol(symbol)
    cs.offset = 0.0
    part.insert(0.0, cs)
    score.append(part)
    return score


def test_ctr_first_beat_single_note_on_downbeat_chord_tone() -> None:
    """Одна нота C ровно на downbeat 0.0 на Cmaj7 → 1.0."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([(60, 0.0, 0.5)])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) == 1.0


def test_ctr_first_beat_only_off_downbeat_returns_none() -> None:
    """Все ноты вдалеке от downbeat'ов — после фильтра ноль нот → None."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([
        (60, 0.5, 0.25),  # между 0.0 и 2.0
        (64, 1.0, 0.25),
        (67, 1.5, 0.25),
        (60, 2.5, 0.25),
        (64, 3.0, 0.25),
    ])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) is None


def test_ctr_first_beat_one_on_downbeat_others_between() -> None:
    """Одна нота C на downbeat 0.0 (chord tone), две ноты вне → фильтрация
    оставляет только первую → 1.0."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([
        (60, 0.0, 0.25),  # downbeat, chord tone
        (61, 0.5, 0.25),  # off-downbeat, non-chord (отфильтруется)
        (66, 1.0, 0.25),  # off-downbeat, non-chord (отфильтруется)
    ])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) == 1.0


def test_ctr_first_beat_two_on_downbeats_half_chord_tones() -> None:
    """C на downbeat 0.0 (chord tone) + C# на downbeat 2.0 (non-chord)
    → после фильтра 2 ноты, 1 хит → 0.5."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([
        (60, 0.0, 0.25),  # downbeat, chord tone
        (61, 2.0, 0.25),  # downbeat, non-chord tone
    ])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) == 0.5


def test_ctr_first_beat_all_on_downbeats_all_chord_tones() -> None:
    """3 ноты на downbeats 0.0/2.0/4.0, все chord tones Cmaj7 → 1.0."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([
        (60, 0.0, 0.25),
        (64, 2.0, 0.25),
        (67, 4.0, 0.25),
    ])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) == 1.0


def test_ctr_first_beat_within_tolerance_counts_as_downbeat() -> None:
    """Нота на 0.02 s (внутри tolerance 0.03125 s) считается на downbeat."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([(60, 0.02, 0.25)])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) == 1.0


def test_ctr_first_beat_just_outside_tolerance_filtered() -> None:
    """Нота на 0.05 s (вне tolerance 0.03125 s) отфильтровывается."""
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([(60, 0.05, 0.25)])
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) is None


def test_ctr_first_beat_zero_notes_returns_none() -> None:
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    ctx = SegmentContext(segment=pm, chord_context=_score_with_chord("Cmaj7"))
    assert CtrFirstBeat().compute(ctx) is None


def test_ctr_first_beat_no_chord_context_returns_none() -> None:
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([(60, 0.0, 0.5)])
    ctx = SegmentContext(segment=pm, chord_context=None)
    assert CtrFirstBeat().compute(ctx) is None


def test_ctr_first_beat_empty_chord_context_returns_none() -> None:
    from base import SegmentContext
    from ctr_first_beat import CtrFirstBeat
    pm = _pm_with_notes([(60, 0.0, 0.5)])
    empty = m21.stream.Score()
    empty.append(m21.stream.Part())
    ctx = SegmentContext(segment=pm, chord_context=empty)
    assert CtrFirstBeat().compute(ctx) is None


def test_ctr_first_beat_name() -> None:
    from ctr_first_beat import CtrFirstBeat
    assert CtrFirstBeat().name == "ctr_first_beat"
