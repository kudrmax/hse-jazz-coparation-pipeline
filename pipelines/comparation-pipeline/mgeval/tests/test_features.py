"""Тесты feature extraction — порт mgeval/core.py."""
import numpy as np
import pretty_midi
import pytest

from mgeval.features import (
    FEATURES, NLH_BINS_RATIO, _bar_length_seconds, _get_sorted_notes,
    avg_ioi, avg_pitch_interval, note_length_hist, note_length_transition_matrix,
    pitch_class_transition_matrix, pitch_range,
    total_pitch_class_histogram, total_used_note, total_used_pitch,
)
from mgeval.tests._helpers import mk_pm


def test_total_used_pitch_happy():
    pm = mk_pm([(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5), (60, 1.5, 2.0)])
    result = total_used_pitch(pm)
    np.testing.assert_array_equal(result, [3])


def test_total_used_pitch_single_note():
    pm = mk_pm([(60, 0.0, 0.5)])
    np.testing.assert_array_equal(total_used_pitch(pm), [1])


def test_total_used_pitch_zero_notes_returns_none():
    pm = mk_pm([])
    assert total_used_pitch(pm) is None


def test_total_used_note_happy():
    pm = mk_pm([(60, 0.0, 0.5), (60, 0.5, 1.0), (64, 1.0, 1.5), (67, 1.5, 2.0)])
    np.testing.assert_array_equal(total_used_note(pm), [4])


def test_total_used_note_zero_returns_none():
    pm = mk_pm([])
    assert total_used_note(pm) is None


def test_pitch_range_happy():
    pm = mk_pm([(60, 0.0, 0.5), (67, 0.5, 1.0), (72, 1.0, 1.5)])
    np.testing.assert_array_equal(pitch_range(pm), [12])  # 72-60


def test_pitch_range_single_note():
    pm = mk_pm([(60, 0.0, 0.5)])
    np.testing.assert_array_equal(pitch_range(pm), [0])


def test_pitch_range_zero_notes_returns_none():
    pm = mk_pm([])
    assert pitch_range(pm) is None


def test_pch_happy_two_pitches_equal_duration():
    # C(60) и E(64), оба длиной 1.0с → pch[0]≈0.5, pch[4]≈0.5
    pm = mk_pm([(60, 0.0, 1.0), (64, 1.0, 2.0)])
    pch = total_pitch_class_histogram(pm)
    assert pch.shape == (12,)
    np.testing.assert_allclose(pch[0], 0.5, atol=0.05)
    np.testing.assert_allclose(pch[4], 0.5, atol=0.05)
    # all other = 0
    expected_zero = [pch[i] for i in range(12) if i not in (0, 4)]
    np.testing.assert_allclose(expected_zero, [0.0] * 10, atol=0.05)


def test_pch_octave_invariance():
    # C3(48) и C5(72), оба → pch[0] = 1.0
    pm = mk_pm([(48, 0.0, 1.0), (72, 1.0, 2.0)])
    pch = total_pitch_class_histogram(pm)
    np.testing.assert_allclose(pch[0], 1.0, atol=0.05)
    expected_zero = [pch[i] for i in range(12) if i != 0]
    np.testing.assert_allclose(expected_zero, [0.0] * 11, atol=0.05)


def test_pch_zero_notes_returns_none():
    pm = mk_pm([])
    assert total_pitch_class_histogram(pm) is None


def test_pctm_happy_c_to_e_to_g():
    # C(60) → E(64) → G(67) — transitions: 0→4, 4→7
    pm = mk_pm([(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5)])
    flat = pitch_class_transition_matrix(pm)
    assert flat.shape == (144,)
    matrix = flat.reshape(12, 12)
    # pretty_midi.get_pitch_class_transition_matrix считает overlap-based
    # transitions — sanity: матрица не нулевая.
    assert matrix.sum() > 0


def test_pctm_zero_notes_returns_none():
    pm = mk_pm([])
    assert pitch_class_transition_matrix(pm) is None


def test_pctm_single_note_returns_none():
    pm = mk_pm([(60, 0.0, 0.5)])
    assert pitch_class_transition_matrix(pm) is None


def test_avg_pitch_interval_happy():
    # pitches=[60, 64, 60] → intervals=[4, -4] → |abs|=[4, 4] → mean=4.0
    pm = mk_pm([(60, 0.0, 0.5), (64, 0.5, 1.0), (60, 1.0, 1.5)])
    np.testing.assert_array_equal(avg_pitch_interval(pm), [4.0])


def test_avg_pitch_interval_unsigned_mean():
    # pitches=[60, 67] → interval=7 → mean=7.0
    pm = mk_pm([(60, 0.0, 0.5), (67, 0.5, 1.0)])
    np.testing.assert_array_equal(avg_pitch_interval(pm), [7.0])


def test_avg_pitch_interval_single_note_returns_none():
    pm = mk_pm([(60, 0.0, 0.5)])
    assert avg_pitch_interval(pm) is None


def test_avg_pitch_interval_zero_notes_returns_none():
    pm = mk_pm([])
    assert avg_pitch_interval(pm) is None


def test_avg_ioi_happy():
    # onsets=[0.0, 0.5, 1.5] → diffs=[0.5, 1.0] → mean=0.75
    pm = mk_pm([(60, 0.0, 0.4), (64, 0.5, 0.9), (67, 1.5, 2.0)])
    np.testing.assert_allclose(avg_ioi(pm), [0.75])


def test_avg_ioi_single_note_returns_none():
    pm = mk_pm([(60, 0.0, 0.5)])
    assert avg_ioi(pm) is None


def test_avg_ioi_zero_notes_returns_none():
    pm = mk_pm([])
    assert avg_ioi(pm) is None


def test_nlh_quarter_only_at_60bpm():
    # 4/4 at 60 BPM: 1 bar = 4.0 sec. unit = 4/96 = 0.0417 sec.
    # quarter = unit * 24 = 1.0 sec. Так что 4 ноты по 1.0с → все в bin[2] (quarter).
    pm = mk_pm(
        [(60, 0.0, 1.0), (62, 1.0, 2.0), (64, 2.0, 3.0), (65, 3.0, 4.0)],
        tempo=60.0,
    )
    nlh = note_length_hist(pm)
    assert nlh.shape == (12,)
    np.testing.assert_allclose(nlh[2], 1.0)
    expected_zero = [nlh[i] for i in range(12) if i != 2]
    np.testing.assert_allclose(expected_zero, [0.0] * 11)


def test_nlh_mixed_durations_at_60bpm():
    # 4/4 at 60: 1 bar=4s. unit=4/96. half=unit*48=2.0s, quarter=1.0s, 8th=0.5s.
    # 1 half + 2 quarters + 4 eighths = 7 нот → hist[1]=1/7, hist[2]=2/7, hist[3]=4/7.
    pm = mk_pm(
        [(60, 0.0, 2.0),  # half
         (60, 2.0, 3.0), (62, 3.0, 4.0),  # 2 quarters
         (64, 4.0, 4.5), (65, 4.5, 5.0), (67, 5.0, 5.5), (69, 5.5, 6.0)],  # 4 eighths
        tempo=60.0,
    )
    nlh = note_length_hist(pm)
    np.testing.assert_allclose(nlh[1], 1 / 7, atol=1e-6)
    np.testing.assert_allclose(nlh[2], 2 / 7, atol=1e-6)
    np.testing.assert_allclose(nlh[3], 4 / 7, atol=1e-6)


def test_nlh_zero_notes_returns_none():
    pm = mk_pm([])
    assert note_length_hist(pm) is None


def test_nlh_bins_ratio_order():
    # Sanity: bins в порядке, заявленном в спеке.
    assert NLH_BINS_RATIO == [96, 48, 24, 12, 6, 72, 36, 18, 9, 32, 16, 8]


def test_nltm_happy_three_quarters_at_60bpm():
    # 3 quarter-notes подряд → 2 перехода quarter→quarter → matrix[2,2]=2.
    pm = mk_pm(
        [(60, 0.0, 1.0), (62, 1.0, 2.0), (64, 2.0, 3.0)],
        tempo=60.0,
    )
    flat = note_length_transition_matrix(pm)
    assert flat.shape == (144,)
    matrix = flat.reshape(12, 12)
    assert matrix[2, 2] == 2.0
    # все остальные нули
    matrix[2, 2] = 0
    assert matrix.sum() == 0


def test_nltm_zero_notes_returns_none():
    pm = mk_pm([])
    assert note_length_transition_matrix(pm) is None


def test_nltm_single_note_returns_none():
    pm = mk_pm([(60, 0.0, 1.0)], tempo=60.0)
    assert note_length_transition_matrix(pm) is None


def test_features_registry_has_nine_in_order():
    expected_order = [
        "total_used_pitch",
        "total_pitch_class_histogram",
        "pitch_class_transition_matrix",
        "pitch_range",
        "avg_pitch_interval",
        "total_used_note",
        "avg_ioi",
        "note_length_hist",
        "note_length_transition_matrix",
    ]
    assert list(FEATURES.keys()) == expected_order


def test_features_registry_callable_on_simple_pm():
    pm = mk_pm([(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5)])
    for name, fn in FEATURES.items():
        out = fn(pm)
        assert out is not None, f"{name} unexpectedly returned None"
        assert isinstance(out, np.ndarray), f"{name} returned {type(out)}"


def test_get_sorted_notes_flattens_instruments():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst1 = pretty_midi.Instrument(program=0)
    inst1.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=1.0, end=1.5))
    inst2 = pretty_midi.Instrument(program=0)
    inst2.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5))
    pm.instruments.extend([inst1, inst2])
    notes = _get_sorted_notes(pm)
    assert [n.pitch for n in notes] == [64, 60]  # sorted by start


def test_get_sorted_notes_empty():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    assert _get_sorted_notes(pm) == []


def test_bar_length_seconds_default_4_4_120bpm():
    pm = mk_pm([(60, 0.0, 1.0)], tempo=120.0)
    # 4/4 at 120 BPM: 60/120 * 4 * (4/4) = 2.0 sec/bar
    assert _bar_length_seconds(pm) == pytest.approx(2.0)


def test_bar_length_seconds_3_4_60bpm():
    pm = mk_pm([(60, 0.0, 1.0)], tempo=60.0, ts_numerator=3, ts_denominator=4)
    # 60/60 * 3 * (4/4) = 3.0 sec/bar
    assert _bar_length_seconds(pm) == pytest.approx(3.0)


def test_bar_length_seconds_no_ts_default_4_4():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    # no time_signature_changes → default 4/4
    assert _bar_length_seconds(pm) == pytest.approx(2.0)
