"""Tests for extract_chord_track: chord-tensor construction (level B unit tests).

Round-trip equivalence with the authors' preprocess is in Task 7."""
import numpy as np

from models.cmt.xml_to_cmt_tensors import extract_chord_track
from helpers import build_test_stream, build_parallel_two_track_midi


# ---------- level B: edge-cases ----------

def test_chord_at_offset_zero_fills_first_frame():
    stream = build_test_stream(
        melody_notes=[],
        chord_changes=[(0.0, "C")],   # C major triad: pitch-classes 0, 4, 7
        num_bars=4,
    )
    arr = extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)
    # First half = theme (2 bars × 16 frames = 32). Second half = same cycled.
    # Total length = num_bars * frame_per_bar + 1 = 65 (extra terminal frame).
    assert arr.shape == (65, 12)
    expected = np.zeros(12)
    expected[[0, 4, 7]] = 1.0
    np.testing.assert_array_equal(arr[0], expected)


def test_chord_held_until_next_change():
    stream = build_test_stream(
        melody_notes=[],
        chord_changes=[(0.0, "C"), (4.0, "G")],   # C for first bar, G from bar 2
        num_bars=4,
    )
    arr = extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)
    # Bar 1 (frames 0..15) = C; bar 2 (frames 16..31) = G.
    expected_c = np.zeros(12); expected_c[[0, 4, 7]] = 1.0
    expected_g = np.zeros(12); expected_g[[7, 11, 2]] = 1.0  # G B D
    np.testing.assert_array_equal(arr[0], expected_c)
    np.testing.assert_array_equal(arr[15], expected_c)
    np.testing.assert_array_equal(arr[16], expected_g)
    np.testing.assert_array_equal(arr[31], expected_g)


def test_no_chord_before_first_chord_symbol_is_zero_vector():
    stream = build_test_stream(
        melody_notes=[],
        chord_changes=[(2.0, "C")],   # ChordSymbol arrives at quarter offset 2
        num_bars=4,
    )
    arr = extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)
    # offset 2 quarters * (16/4) frames-per-quarter = frame 8. Frames 0..7 = zeros.
    np.testing.assert_array_equal(arr[0], np.zeros(12))
    np.testing.assert_array_equal(arr[7], np.zeros(12))
    expected = np.zeros(12); expected[[0, 4, 7]] = 1.0
    np.testing.assert_array_equal(arr[8], expected)


def test_slash_chord_keeps_all_pitch_classes():
    # Cmaj7/G — bass is G; preprocess.py would drop it via [1:] but we don't.
    stream = build_test_stream(
        melody_notes=[],
        chord_changes=[(0.0, "Cmaj7/G")],   # Cmaj7 = C E G B; slash bass = G
        num_bars=4,
    )
    arr = extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)
    # All four pitch-classes (C=0, E=4, G=7, B=11) should be set.
    nonzero = sorted(arr[0].nonzero()[0].tolist())
    assert nonzero == [0, 4, 7, 11]


def test_second_half_cycles_first_half():
    stream = build_test_stream(
        melody_notes=[],
        chord_changes=[(0.0, "C"), (4.0, "G")],   # 2-bar prog: C, G
        num_bars=4,
    )
    arr = extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)
    # Second half (frames 32..63) = cycled copy of first half (0..31); frame 64 = cyclic too.
    np.testing.assert_array_equal(arr[32:64], arr[0:32])


def test_unrecognized_chord_symbol_raises():
    import pytest
    # music21 raises ValueError already at ChordSymbol("WTFchord") construction
    # (inside build_test_stream), so we wrap the whole setup+call together.
    # Either way, the pipeline raises ValueError on bad chord figures.
    with pytest.raises(ValueError):
        stream = build_test_stream(
            melody_notes=[],
            chord_changes=[(0.0, "WTFchord")],
            num_bars=4,
        )
        extract_chord_track(stream, num_bars=4, theme_bars=2, frame_per_bar=16)


# ---------- level A: round-trip with authors' preprocess ----------

def test_chord_track_matches_authors_preprocess_for_simple_progression(tmp_path):
    """Build the same content as a music21 stream AND a 2-track MIDI, then
    check that our extract_chord_track agrees with the chord column from
    preprocess.extract_instances_from_midi for the theme-half frames.
    """
    import preprocess

    # 8 bars of music; theme = 4 bars (C G Am F); second half is the cycled repeat.
    # stream gets only the theme-half chord_changes so extract_chord_track cycles them.
    # MIDI gets chord_changes for all 8 bars (two choruses) so preprocess sees the same
    # cycled progression that extract_chord_track produces.
    theme_chord_changes = [(0.0, "C"), (4.0, "G"), (8.0, "Am"), (12.0, "F")]
    all_chord_changes = theme_chord_changes + [
        (off + 16.0, fig) for off, fig in theme_chord_changes
    ]
    # Melody constraints to survive preprocess filters:
    # • >5 unique pitches (variety filter)
    # • All within one octave so consecutive jumps ≤12 semitones (cont-rest filter)
    # • 72 eighth-notes = 36 quarters → 18 s @ 120 bpm → 144 frames @ 8 fps > 129
    #   (timelen >= instance_len + 2 needed for the sliding-window loop to fire)
    scale_pitches = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C5",
                     "D5", "E5", "F5", "G5", "A5", "B5", "C5", "D5"]
    melody_notes = [
        (i * 0.5, 0.5, scale_pitches[i % len(scale_pitches)])
        for i in range(72)
    ]

    stream = build_test_stream(
        melody_notes=melody_notes,
        chord_changes=theme_chord_changes,
        num_bars=8,
    )
    midi = build_parallel_two_track_midi(
        melody_notes=melody_notes,
        chord_changes=all_chord_changes,
        num_bars=8,
    )

    midi_path = tmp_path / "rt.mid"
    midi.write(str(midi_path))

    ours = extract_chord_track(stream, num_bars=8, theme_bars=4, frame_per_bar=16)

    # Authors' instances each have shape (instance_len + 1, 12) == (129, 12);
    # we compare only the first instance's first num_bars*frame_per_bar = 128 frames.
    instances = preprocess.extract_instances_from_midi(
        str(midi_path),
        num_bars=8, frame_per_bar=16, stride_bars=4, pitch_range=48,
    )
    assert len(instances) >= 1, "preprocess returned no windows — check filters"
    theirs_sparse = instances[0]["chord"]
    theirs = theirs_sparse.toarray()[: 8 * 16]   # (128, 12)

    np.testing.assert_array_equal(ours[: 8 * 16], theirs)
