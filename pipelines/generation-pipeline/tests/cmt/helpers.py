"""Shared test helpers for the CMT wrapper test suite.

Importable directly as `from helpers import build_test_stream` inside any test
under tests/cmt/ — avoids name collision with models/CMT-pytorch/conftest.py
which pytest picks up first when resolving a bare `from conftest import`.
"""
from __future__ import annotations

import music21 as m21
import pretty_midi


def build_test_stream(
    melody_notes: list[tuple[float, float, str | None]],
    chord_changes: list[tuple[float, str]],
    num_bars: int,
) -> m21.stream.Stream:
    """Build a music21 Stream with melody notes + ChordSymbol harmony.

    melody_notes: list of (offset_in_quarters, duration_in_quarters, pitch_name_or_None_for_rest).
    chord_changes: list of (offset_in_quarters, figure) — e.g. (0.0, "Cmaj7").
    num_bars: total bars (for measure structure; 4/4 only).

    Returns a single-part Stream with measures, in C major (no transposition needed).
    """
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.key.Key("C", "major"))
    for bar in range(num_bars):
        part.append(m21.stream.Measure(number=bar + 1))

    for offset_q, figure in chord_changes:
        part.insert(offset_q, m21.harmony.ChordSymbol(figure))

    for offset_q, dur_q, pitch_name in melody_notes:
        if pitch_name is None:
            elem = m21.note.Rest(quarterLength=dur_q)
        else:
            elem = m21.note.Note(pitch_name, quarterLength=dur_q)
        part.insert(offset_q, elem)

    score = m21.stream.Score()
    score.append(part)
    return score


def build_parallel_two_track_midi(
    melody_notes: list[tuple[float, float, str | None]],
    chord_changes: list[tuple[float, str]],
    num_bars: int,
    bpm: float = 120.0,
) -> pretty_midi.PrettyMIDI:
    """Build a 2-track MIDI mirroring the same content as build_test_stream,
    in the format preprocess.extract_instances_from_midi consumes.

    Track 0: melody (single voice, full durations).
    Track 1: chord — at each chord change, emit bass = lowest pitch ONE OCTAVE
             DOWN + chord tones from ChordSymbol(figure).pitches at that offset.

    Times are derived from quarterLength * (60/bpm).
    """
    seconds_per_quarter = 60.0 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    melody = pretty_midi.Instrument(program=0, name="melody")
    chord_inst = pretty_midi.Instrument(program=0, name="chord")

    for offset_q, dur_q, pitch_name in melody_notes:
        if pitch_name is None:
            continue
        start = offset_q * seconds_per_quarter
        end = (offset_q + dur_q) * seconds_per_quarter
        midi_pitch = m21.pitch.Pitch(pitch_name).midi
        melody.notes.append(
            pretty_midi.Note(velocity=100, pitch=midi_pitch, start=start, end=end)
        )

    chord_dur_placeholder = 0.5
    for offset_q, figure in chord_changes:
        cs = m21.harmony.ChordSymbol(figure)
        midis = sorted(p.midi for p in cs.pitches)
        # Emit bass-octave-down + chord tones, mirroring how the original
        # cleansed_midi_twotrack_ckey dataset is structured.
        bass_midi = midis[0] - 12
        start = offset_q * seconds_per_quarter
        end = start + chord_dur_placeholder
        chord_inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=bass_midi, start=start, end=end)
        )
        for p in midis:
            chord_inst.notes.append(
                pretty_midi.Note(velocity=100, pitch=p, start=start, end=end)
            )

    pm.instruments.append(melody)
    pm.instruments.append(chord_inst)
    return pm
