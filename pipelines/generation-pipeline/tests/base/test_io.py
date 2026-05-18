"""Field shape + behavior tests for BaseGeneratorInput / BaseGeneratorOutput."""
from pathlib import Path

import pretty_midi


def test_input_dataclass_required_fields():
    from models.base.io import BaseGeneratorInput

    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    assert inp.musicxml_path == Path("/tmp/x.xml")
    assert inp.seed == 1
    assert inp.input_bars == 8
    assert inp.output_bars == 8


def test_output_save_midi_creates_parent_dirs(tmp_path):
    from models.base.io import BaseGeneratorOutput

    pm = pretty_midi.PrettyMIDI()
    out = BaseGeneratorOutput(
        midi=pm, title="x", seed=1, input_bars=8, output_bars=8, inference_time=0.0
    )
    target = tmp_path / "deeply" / "nested" / "out.mid"
    saved = out.save_midi(target)
    assert saved == target
    assert target.is_file()


def test_output_get_midi_returns_same_object():
    from models.base.io import BaseGeneratorOutput

    pm = pretty_midi.PrettyMIDI()
    out = BaseGeneratorOutput(
        midi=pm, title="x", seed=1, input_bars=8, output_bars=8, inference_time=0.0
    )
    assert out.get_midi() is pm
