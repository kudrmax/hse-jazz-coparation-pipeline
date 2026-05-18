"""Sanity-check that constants match the contract baked into the CMT fork."""
from models.cmt import constants as C


def test_rhythm_tokens_match_preprocess_semantics():
    # preprocess.py uses these literal values when building rhythm_idx.
    assert C.RHYTHM_REST == 0
    assert C.RHYTHM_HOLD == 1
    assert C.RHYTHM_ONSET == 2


def test_chord_pitch_classes_is_octave():
    assert C.CHORD_PITCH_CLASSES == 12


def test_melody_instrument_index_matches_pitch_to_midi_contract():
    # utils/utils.py:pitch_to_midi appends melody first, chord second.
    assert C.MELODY_INSTRUMENT_INDEX == 0


def test_beats_per_bar_is_four_four_assumption():
    assert C.BEATS_PER_BAR == 4


def test_target_keys():
    assert C.TARGET_KEY_MAJOR == "C"
    assert C.TARGET_KEY_MINOR == "A"
