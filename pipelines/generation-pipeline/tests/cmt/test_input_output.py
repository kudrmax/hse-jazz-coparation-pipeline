"""Shape and default checks for GeneratorCmtInput / GeneratorCmtOutput."""
from pathlib import Path

import pretty_midi


def test_input_dataclass_fields_and_defaults():
    from models.cmt.input import GeneratorCmtInput

    inp = GeneratorCmtInput(
        musicxml_path=Path("/tmp/x.xml"),
        seed=1,
        input_bars=8,
        output_bars=8,
    )
    assert inp.musicxml_path == Path("/tmp/x.xml")
    assert inp.seed == 1
    assert inp.input_bars == 8
    assert inp.output_bars == 8
    assert inp.topk == 5  # default
    assert inp.get_musicxml_path() == Path("/tmp/x.xml")


def test_input_topk_overridable():
    from models.cmt.input import GeneratorCmtInput
    inp = GeneratorCmtInput(
        musicxml_path=Path("/tmp/x.xml"),
        seed=2, input_bars=8, output_bars=8, topk=10,
    )
    assert inp.topk == 10


def test_output_dataclass_fields_and_helpers(tmp_path):
    from models.cmt.output import GeneratorCmtOutput

    pm = pretty_midi.PrettyMIDI()
    out = GeneratorCmtOutput(
        midi=pm,
        title="x",
        seed=1,
        input_bars=8,
        output_bars=8,
        inference_time=0.0,
        num_bars=16,
        frame_per_bar=16,
        topk=5,
        checkpoint_epoch=42,
        transpose_semitones=-2,
    )
    assert out.transpose_semitones == -2
    assert out.get_midi() is pm
    target = tmp_path / "subdir" / "out.mid"
    saved = out.save_midi(target)
    assert saved == target
    assert target.is_file()


def test_output_accepts_none_checkpoint_epoch():
    from models.cmt.output import GeneratorCmtOutput
    pm = pretty_midi.PrettyMIDI()
    out = GeneratorCmtOutput(
        midi=pm, title="x", seed=1, input_bars=8, output_bars=8,
        inference_time=0.0, num_bars=16, frame_per_bar=16, topk=5,
        checkpoint_epoch=None,
    )
    assert out.checkpoint_epoch is None


def test_output_transpose_semitones_defaults_to_zero():
    from models.cmt.output import GeneratorCmtOutput
    pm = pretty_midi.PrettyMIDI()
    out = GeneratorCmtOutput(
        midi=pm, title="x", seed=1, input_bars=8, output_bars=8,
        inference_time=0.0, num_bars=16, frame_per_bar=16, topk=5,
        checkpoint_epoch=None,
    )
    assert out.transpose_semitones == 0
