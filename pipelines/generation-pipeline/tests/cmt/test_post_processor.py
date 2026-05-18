"""Tests for CmtPostProcessor — runs super() (key_signature) + transpose_back."""
from pathlib import Path

import music21 as m21
import pretty_midi

from models.base.io import BaseGeneratorInput
from models.cmt.output import GeneratorCmtOutput
from models.cmt.post_processor import CmtPostProcessor


def _build_a_minor_stream(n_bars: int = 8) -> m21.stream.Score:
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.key.Key("A", "minor"))
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        for pitch in ("A4", "C5", "E5", "B4"):
            m.append(m21.note.Note(pitch, quarterLength=1.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    return score


def _midi_with_two_notes() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
    pm.instruments.append(inst)
    return pm


def test_super_sets_key_signature_and_transpose_shifts_notes():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = GeneratorCmtOutput(
        midi=_midi_with_two_notes(),
        title="x", seed=1, input_bars=8, output_bars=8, inference_time=0.0,
        num_bars=16, frame_per_bar=16, topk=5, checkpoint_epoch=None,
        transpose_semitones=-2,
    )
    CmtPostProcessor().process(inp, stream, out)

    # super() effect: key_signature_changes populated.
    assert len(out.midi.key_signature_changes) == 1
    assert out.midi.key_signature_changes[0].key_number == 21  # a minor

    # CMT effect: every note pitch shifted by -2.
    assert out.midi.instruments[0].notes[0].pitch == 58
    assert out.midi.instruments[0].notes[1].pitch == 62


def test_zero_semitones_is_noop_for_pitches():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = GeneratorCmtOutput(
        midi=_midi_with_two_notes(),
        title="x", seed=1, input_bars=8, output_bars=8, inference_time=0.0,
        num_bars=16, frame_per_bar=16, topk=5, checkpoint_epoch=None,
        transpose_semitones=0,
    )
    CmtPostProcessor().process(inp, stream, out)

    # key_signature still set.
    assert len(out.midi.key_signature_changes) == 1
    # pitches untouched.
    assert out.midi.instruments[0].notes[0].pitch == 60
    assert out.midi.instruments[0].notes[1].pitch == 64
