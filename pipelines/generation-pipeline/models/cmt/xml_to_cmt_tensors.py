"""Step 3 of GeneratorCmt.generate(): musicxml stream -> three CMT tensors.

Public surface:
- extract_chord_track : 12-dim multi-hot pitch-class vector per frame.
- extract_melody_tracks : (rhythm, pitch, base_note) for the theme half only.
- convert : convenience wrapper that calls both and returns a dict.

The chord track is filled for the FULL window (num_bars * frame_per_bar
frames), with the second half being a cyclic copy of the first half. This
mirrors how the MINGUS/BebopNet wrappers feed chord progressions on the
generated half (see music_generator.py:206 and gen_funct.py:275 in the
respective forks).

The melody tracks (rhythm, pitch) are filled for the THEME half only —
model.sampling pads the prime to max_len internally.
"""
from __future__ import annotations

import music21 as m21
import numpy as np

from .constants import (
    BEATS_PER_BAR,
    CHORD_PITCH_CLASSES,
    RHYTHM_HOLD,
    RHYTHM_ONSET,
    RHYTHM_REST,
)


def _frames_per_quarter(frame_per_bar: int) -> float:
    return frame_per_bar / BEATS_PER_BAR


def extract_chord_track(
    stream: m21.stream.Stream,
    num_bars: int,
    theme_bars: int,
    frame_per_bar: int,
) -> np.ndarray:
    """Build the chord-tensor (num_bars * frame_per_bar + 1, 12) multi-hot float32.

    Theme half (first theme_bars * frame_per_bar frames) is filled from
    ChordSymbol elements in the stream; remaining frames (including the extra
    terminal frame) are a cyclic copy of the theme half.

    The trailing extra frame matches the authors' format
    (``preprocess.extract_instances_from_midi`` produces ``instance_len + 1``
    rows; ``model.sampling`` consumes the full length, while ``pitch_to_midi``
    is called with the leading ``instance_len`` rows).
    """
    theme_steps = theme_bars * frame_per_bar
    total_steps = num_bars * frame_per_bar
    fpq = _frames_per_quarter(frame_per_bar)

    chord_symbols = list(stream.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    # Resolve pitch-classes eagerly so any unrecognized figure raises here, not later.
    resolved: list[tuple[int, list[int]]] = []
    for cs in chord_symbols:
        pitches = cs.pitches
        if not pitches:
            raise ValueError(f"ChordSymbol {cs.figure!r} resolved to no pitches")
        offset_quarters = float(cs.getOffsetInHierarchy(stream))
        frame = int(round(offset_quarters * fpq))
        pitch_classes = sorted({p.midi % 12 for p in pitches})
        resolved.append((frame, pitch_classes))

    arr = np.zeros((total_steps + 1, CHORD_PITCH_CLASSES), dtype=np.float32)
    prev_chord = np.zeros(CHORD_PITCH_CLASSES, dtype=np.float32)

    change_map: dict[int, np.ndarray] = {}
    for frame, pcs in resolved:
        if 0 <= frame < theme_steps:
            vec = np.zeros(CHORD_PITCH_CLASSES, dtype=np.float32)
            for pc in pcs:
                vec[pc] = 1.0
            change_map[frame] = vec

    for f in range(theme_steps):
        if f in change_map:
            prev_chord = change_map[f]
        arr[f] = prev_chord

    # Cycle theme half over the remaining frames, including the extra terminal frame.
    for f in range(theme_steps, total_steps + 1):
        arr[f] = arr[f - theme_steps]

    return arr


def extract_melody_tracks(
    stream: m21.stream.Stream,
    theme_bars: int,
    frame_per_bar: int,
    pitch_range: int = 48,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build (rhythm, pitch, base_note) for the theme half (length =
    theme_bars * frame_per_bar). The model itself extends this prime to the
    full window during sampling.

    pitch_range = 48 mirrors the authors' configuration (num_pitch=50 minus
    the two reserved tokens hold=48 and rest=49).
    """
    theme_steps = theme_bars * frame_per_bar
    fpq = _frames_per_quarter(frame_per_bar)

    # Merge notes joined by ties (musicxml <tie>/<tied>) before quantizing
    # so a sustained note across a bar line stays a single onset, not two.
    # stripTies() returns a copy; does not affect ChordSymbol elements.
    merged = stream.stripTies()
    # Collect every Note (skip rests; they fall through as the default RHYTHM_REST).
    notes = []
    for n in merged.recurse().getElementsByClass(m21.note.Note):
        offset_quarters = float(n.getOffsetInHierarchy(merged))
        notes.append((offset_quarters, float(n.quarterLength), n.pitch.midi))

    if not notes:
        raise ValueError("Theme contains no notes")

    base_note = 12 * (min(p for _, _, p in notes) // 12)
    pitch_hold = pitch_range
    pitch_rest = pitch_range + 1

    rhythm = np.full(theme_steps, RHYTHM_REST, dtype=np.int64)
    pitch = np.full(theme_steps, pitch_rest, dtype=np.int64)

    for offset_q, dur_q, midi in notes:
        rel = midi - base_note
        if not (0 <= rel < pitch_range):
            raise ValueError(
                f"Note MIDI {midi} is {rel} semitones above base_note {base_note}; "
                f"exceeds pitch_range={pitch_range}. Theme spans more than {pitch_range} semitones."
            )
        f_start = int(round(offset_q * fpq))
        f_count = max(1, int(round(dur_q * fpq)))
        f_end = min(f_start + f_count, theme_steps)
        if f_start >= theme_steps:
            continue
        rhythm[f_start] = RHYTHM_ONSET
        pitch[f_start] = rel
        for f in range(f_start + 1, f_end):
            rhythm[f] = RHYTHM_HOLD
            pitch[f] = pitch_hold

    return rhythm, pitch, int(base_note)


def convert(
    stream: m21.stream.Stream,
    num_bars: int,
    theme_bars: int,
    frame_per_bar: int,
    pitch_range: int = 48,
) -> dict:
    """Build all three CMT tensors plus base_note from a music21 stream."""
    rhythm, pitch, base_note = extract_melody_tracks(
        stream, theme_bars=theme_bars, frame_per_bar=frame_per_bar,
        pitch_range=pitch_range,
    )
    chord = extract_chord_track(
        stream,
        num_bars=num_bars,
        theme_bars=theme_bars,
        frame_per_bar=frame_per_bar,
    )
    return {"rhythm": rhythm, "pitch": pitch, "chord": chord, "base_note": base_note}
