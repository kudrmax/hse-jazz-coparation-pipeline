"""Tests for extract_melody_tracks: rhythm + pitch + base_note (level B)."""
import numpy as np

from models.cmt.xml_to_cmt_tensors import extract_melody_tracks
from helpers import build_test_stream


def test_single_quarter_note_has_one_onset_then_three_holds():
    # frame_per_bar=16 → 4 frames per quarter. One quarter at C5 (MIDI 72).
    stream = build_test_stream(
        melody_notes=[(0.0, 1.0, "C5")],
        chord_changes=[(0.0, "C")],
        num_bars=2,
    )
    rhythm, pitch, base_note = extract_melody_tracks(
        stream, theme_bars=1, frame_per_bar=16
    )
    # base_note rounds C5 (72) down to nearest octave → 72.
    assert base_note == 72
    # Frame 0: onset; Frames 1, 2, 3: hold; Frames 4..15: rest.
    assert rhythm[0] == 2  # RHYTHM_ONSET
    assert (rhythm[1:4] == 1).all()  # RHYTHM_HOLD
    assert (rhythm[4:16] == 0).all()  # RHYTHM_REST
    assert pitch[0] == 72 - 72  # = 0
    assert (pitch[1:4] == 48).all()  # hold token
    assert (pitch[4:16] == 49).all()  # rest token


def test_rest_at_start_of_bar_yields_rest_tokens():
    stream = build_test_stream(
        melody_notes=[(0.0, 0.5, None), (0.5, 0.5, "C5")],
        chord_changes=[(0.0, "C")],
        num_bars=1,
    )
    rhythm, pitch, _ = extract_melody_tracks(stream, theme_bars=1, frame_per_bar=16)
    # Frames 0, 1 = rest; Frame 2 = onset C5; Frame 3 = hold.
    assert (rhythm[0:2] == 0).all()
    assert rhythm[2] == 2
    assert rhythm[3] == 1
    assert (pitch[0:2] == 49).all()


def test_base_note_rounds_to_nearest_octave_below():
    # Lowest note is E4 (MIDI 64). 12 * (64 // 12) = 60.
    stream = build_test_stream(
        melody_notes=[(0.0, 0.5, "G4"), (0.5, 0.5, "E4"), (1.0, 1.0, "C5")],
        chord_changes=[(0.0, "C")],
        num_bars=1,
    )
    _, _, base_note = extract_melody_tracks(stream, theme_bars=1, frame_per_bar=16)
    assert base_note == 60


def test_eighth_note_triplet_quantizes_to_16th_grid():
    # Three triplet eighths: each duration = 1/3 quarter ≈ 0.333.
    # On 16th-grid (4 frames per quarter), round() yields 1, 1, 2 frames.
    # Pattern: onset at 0, onset at 1, onset at 2, hold at 3.
    stream = build_test_stream(
        melody_notes=[
            (0.0, 1.0 / 3, "C5"),
            (1.0 / 3, 1.0 / 3, "D5"),
            (2.0 / 3, 1.0 / 3, "E5"),
        ],
        chord_changes=[(0.0, "C")],
        num_bars=1,
    )
    rhythm, _, _ = extract_melody_tracks(stream, theme_bars=1, frame_per_bar=16)
    # Three onsets in the first 4 frames, one hold, then rest.
    onset_frames = (rhythm[:4] == 2).sum()
    assert onset_frames == 3, f"expected 3 onsets in first quarter, got {onset_frames}"


def test_pitch_out_of_range_raises():
    # MIDI 120 vs lowest 60 → 60 semitones above base_note > pitch_range=48.
    import pytest
    stream = build_test_stream(
        melody_notes=[(0.0, 0.5, "C5"), (0.5, 0.5, "C9")],   # C9 = MIDI 120
        chord_changes=[(0.0, "C")],
        num_bars=1,
    )
    with pytest.raises(ValueError, match="pitch_range"):
        extract_melody_tracks(stream, theme_bars=1, frame_per_bar=16)


def test_tied_notes_merge_into_single_onset():
    """A note that spans a bar line via <tie> should produce ONE onset
    plus holds, NOT two onsets. Reproduces a real case in
    Autumn_Leaves_8bars.musicxml where F5 (whole) is tied to F5 (quarter).
    """
    import music21 as m21

    # Build a 2-bar stream where bar 1 holds a whole-note F5 tied into
    # bar 2's quarter-note F5 (total: 5 quarters of F5, then 3 quarters rest).
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))

    bar1 = m21.stream.Measure(number=1)
    n1 = m21.note.Note("F5", quarterLength=4.0)
    n1.tie = m21.tie.Tie("start")
    bar1.append(n1)
    part.append(bar1)

    bar2 = m21.stream.Measure(number=2)
    n2 = m21.note.Note("F5", quarterLength=1.0)
    n2.tie = m21.tie.Tie("stop")
    bar2.append(n2)
    bar2.append(m21.note.Rest(quarterLength=3.0))
    part.append(bar2)

    score = m21.stream.Score()
    score.append(part)

    rhythm, pitch, base_note = extract_melody_tracks(
        score, theme_bars=2, frame_per_bar=16
    )
    # frame_per_bar=16 → 4 frames per quarter. 5 quarters = 20 frames.
    # Expect: rhythm[0]=onset, rhythm[1..19]=hold, rhythm[20..31]=rest.
    assert rhythm[0] == 2, f"expected onset at frame 0, got {rhythm[0]}"
    onsets_in_first_5q = (rhythm[:20] == 2).sum()
    assert onsets_in_first_5q == 1, (
        f"tied F5 must produce exactly ONE onset across the bar line, got "
        f"{onsets_in_first_5q} onset(s) in frames 0..19"
    )
    assert (rhythm[1:20] == 1).all(), "frames 1..19 must all be HOLD"
    assert (rhythm[20:32] == 0).all(), "frames 20..31 must all be REST"


# ---------- level A: round-trip with authors' preprocess ----------

def test_melody_tracks_match_authors_preprocess(tmp_path):
    """For an input expressible as both music21 and 2-track MIDI, our
    extract_melody_tracks must agree with the rhythm + pitch columns from
    preprocess.extract_instances_from_midi for the theme half."""
    import preprocess
    from helpers import build_test_stream, build_parallel_two_track_midi

    # 8 bars; melody is a slow C-major scale on quarter notes.
    # Same pattern as chord round-trip (Task 7): 72 eighth-notes across 8 bars,
    # cycling C5..B5 (never crossing C6 boundary → max jump ≤ 11 semitones).
    scale_pitches = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C5",
                     "D5", "E5", "F5", "G5", "A5", "B5", "C5", "D5"]
    melody_notes = [
        (i * 0.5, 0.5, scale_pitches[i % len(scale_pitches)])
        for i in range(72)
    ]

    # Chord: same C chord for all 8 bars (simple, passes chord-count filter).
    theme_chord_changes = [(0.0, "C")] + [(b * 4.0, "C") for b in range(1, 4)]
    all_chord_changes = [(0.0, "C")] + [(b * 4.0, "C") for b in range(1, 8)]

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
    midi_path = tmp_path / "rt_melody.mid"
    midi.write(str(midi_path))

    rhythm_ours, pitch_ours, base_ours = extract_melody_tracks(
        stream, theme_bars=4, frame_per_bar=16
    )

    instances = preprocess.extract_instances_from_midi(
        str(midi_path),
        num_bars=8, frame_per_bar=16, stride_bars=4, pitch_range=48,
    )
    assert len(instances) >= 1, "preprocess returned no windows — check filters"
    inst = instances[0]
    theirs_rhythm = inst["rhythm"][: 4 * 16]
    theirs_pitch = inst["pitch"][: 4 * 16]
    theirs_base = inst["base_note"]

    np.testing.assert_array_equal(rhythm_ours, theirs_rhythm)
    np.testing.assert_array_equal(pitch_ours, theirs_pitch)
    assert base_ours == theirs_base
