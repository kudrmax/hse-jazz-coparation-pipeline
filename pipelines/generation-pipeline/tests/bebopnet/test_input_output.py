"""Field shape checks for GeneratorBebopnet I/O."""
from pathlib import Path

import pretty_midi


def test_input_fields_and_defaults():
    from models.bebopnet.input import GeneratorBebopnetInput

    inp = GeneratorBebopnetInput(
        musicxml_path=Path("/tmp/x.xml"),
        seed=1, input_bars=33, output_bars=32,
    )
    assert inp.input_bars == 33
    assert inp.output_bars == 32
    assert inp.temperature == 1.0


def test_output_fields(tmp_path):
    from models.bebopnet.output import GeneratorBebopnetOutput

    pm = pretty_midi.PrettyMIDI()
    out = GeneratorBebopnetOutput(
        midi=pm, title="x", seed=1,
        input_bars=33, output_bars=32, inference_time=0.0,
        temperature=1.0, top_likelihood=-3.5,
    )
    assert out.top_likelihood == -3.5
