"""Step 1, 2, 6 of GeneratorCmt.generate().

Step 1 — analyze_key: detect tonic + mode of the input theme.
Step 2 — transpose_to_target: shift the stream into C major or A minor and
         report the semitone offset.
Step 6 — transpose_midi_back: undo the offset on a PrettyMIDI in place.
"""
from __future__ import annotations

import music21 as m21
import pretty_midi

from .constants import TARGET_KEY_MAJOR, TARGET_KEY_MINOR


def analyze_key(stream: m21.stream.Stream) -> m21.key.Key:
    ts_elements = list(stream.recurse().getElementsByClass(m21.meter.TimeSignature))
    if ts_elements and ts_elements[0].ratioString != "4/4":
        raise ValueError(
            f"GeneratorCmt only supports 4/4 themes; got {ts_elements[0].ratioString}"
        )
    return stream.analyze("key")


def transpose_to_target(
    stream: m21.stream.Stream, key: m21.key.Key
) -> tuple[m21.stream.Stream, int]:
    """Shift `stream` so its key becomes C major (mode='major') or A minor
    (mode='minor'). Returns the transposed stream and the semitone offset
    used; pass `-offset` to `transpose_midi_back` afterwards.
    """
    target_tonic = TARGET_KEY_MAJOR if key.mode == "major" else TARGET_KEY_MINOR
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch(target_tonic))
    transposed = stream.transpose(interval)
    return transposed, interval.semitones


def transpose_midi_back(midi: pretty_midi.PrettyMIDI, semitones: int) -> None:
    """Shift every note pitch by `semitones` (in place)."""
    if semitones == 0:
        return
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.pitch += semitones
