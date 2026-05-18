"""GeneratorBebopnet now runs torch inference via subprocess. Init no
longer loads the model or unpickles the converter; that work happens in
forked-venv subprocesses (vocab_dump_runner once at init, _subprocess_runner
on every generate)."""
from __future__ import annotations

from pathlib import Path

import pytest

from models.bebopnet import GeneratorBebopnet, GeneratorBebopnetInput

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BBN_FORK = _REPO_ROOT / "models" / "bebopnet-code"
_BBN_MODEL_DIR = _BBN_FORK / "result" / "paper-default"
BBN_VENV_PY = _BBN_FORK / ".venv" / "bin" / "python"
BBN_FORK = _BBN_FORK
BBN_MODEL_DIR = _BBN_MODEL_DIR
THEME_XML = _REPO_ROOT / "pipelines" / "generation-pipeline" / "inputs" / "musicxml" / "Autumn_Leaves.musicxml"
_INPUT_XML = THEME_XML


@pytest.mark.skipif(not (_BBN_MODEL_DIR / "model_best.pt").is_file(),
                     reason="bebopnet checkpoint missing")
def test_init_does_not_import_torch(tmp_path: Path) -> None:
    """Constructing GeneratorBebopnet must not pull torch into the
    pipeline-venv interpreter — torch only lives inside subprocess."""
    import sys
    sys.modules.pop("torch", None)
    gen = GeneratorBebopnet(
        fork_root=_BBN_FORK,
        model_dir=_BBN_MODEL_DIR,
        checkpoint="model_best.pt",
        device="cpu",
    )
    assert "torch" not in sys.modules, "GeneratorBebopnet.__init__ must not import torch"
    assert gen.fork_root == _BBN_FORK
    assert gen.model_dir == _BBN_MODEL_DIR


@pytest.mark.skipif(not (_BBN_MODEL_DIR / "model_best.pt").is_file(),
                     reason="bebopnet checkpoint missing")
def test_generate_end_to_end_via_subprocess(tmp_path: Path) -> None:
    gen = GeneratorBebopnet(
        fork_root=_BBN_FORK,
        model_dir=_BBN_MODEL_DIR,
        checkpoint="model_best.pt",
        device="cpu",
    )
    inp = GeneratorBebopnetInput(
        musicxml_path=_INPUT_XML,
        seed=1, input_bars=32, output_bars=32, temperature=1.0,
    )
    out = gen.generate(inp)
    assert out.midi.instruments
    assert out.midi.instruments[0].notes
    assert isinstance(out.top_likelihood, float)


@pytest.mark.skipif(not BBN_VENV_PY.exists(), reason="BebopNet venv not available")
def test_generator_reuses_subprocess_across_calls(tmp_path: Path) -> None:
    """Дважды зовём gen.generate(...). Subprocess грузится один раз."""
    gen = GeneratorBebopnet(
        fork_root=BBN_FORK,
        model_dir=BBN_MODEL_DIR,
        checkpoint="model_best.pt",
        device="cpu",
    )
    try:
        out1 = gen.generate(GeneratorBebopnetInput(
            musicxml_path=THEME_XML, seed=1, input_bars=32, output_bars=32, temperature=1.0,
        ))
        out2 = gen.generate(GeneratorBebopnetInput(
            musicxml_path=THEME_XML, seed=2, input_bars=32, output_bars=32, temperature=1.0,
        ))
        assert out1.midi.instruments and out2.midi.instruments
        assert out1.midi.instruments[0].notes != out2.midi.instruments[0].notes
    finally:
        gen.close()
