"""Field shape checks for GeneratorMingusInput / GeneratorMingusOutput."""
from pathlib import Path

import pretty_midi


def test_input_fields_and_defaults():
    from models.mingus.input import GeneratorMingusInput

    inp = GeneratorMingusInput(
        musicxml_path=Path("/tmp/x.xml"),
        seed=1, input_bars=33, output_bars=33,
    )
    assert inp.musicxml_path == Path("/tmp/x.xml")
    assert inp.input_bars == 33
    assert inp.output_bars == 33
    assert inp.temperature == 1.0  # default
    assert inp.get_musicxml_path() == Path("/tmp/x.xml")


def test_output_fields(tmp_path):
    from models.mingus.output import GeneratorMingusOutput

    pm = pretty_midi.PrettyMIDI()
    out = GeneratorMingusOutput(
        midi=pm, title="x", seed=1,
        input_bars=33, output_bars=99, inference_time=0.0,
        tempo=120.0, temperature=1.0, epochs=100, cond="I-C-NC-B-BE-O",
    )
    assert out.epochs == 100
    target = tmp_path / "out.mid"
    out.save_midi(target)
    assert target.is_file()
