"""GeneratorMingus now runs torch inference via persistent subprocess.
Init no longer loads the model; that work happens lazily on first generate()
call inside the forked venv subprocess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from models.mingus import GeneratorMingus, GeneratorMingusInput

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MINGUS_FORK = _REPO_ROOT / "models" / "MINGUS"
_MINGUS_VENV_PY = _MINGUS_FORK / ".venv" / "bin" / "python"
_MINGUS_DATA = _MINGUS_FORK / "A_preprocessData" / "data" / "DATA.json"
_MINGUS_CKPT_DIR = _MINGUS_FORK / "B_train" / "models" / "paper-optimal"
_INPUT_XML = _REPO_ROOT / "pipelines" / "generation-pipeline" / "inputs" / "musicxml" / "Autumn_Leaves.musicxml"


@pytest.mark.skipif(not _MINGUS_VENV_PY.exists(), reason="MINGUS venv not available")
@pytest.mark.skipif(not _MINGUS_DATA.is_file(), reason="MINGUS DATA.json absent")
@pytest.mark.skipif(not _MINGUS_CKPT_DIR.is_dir(), reason="MINGUS checkpoints absent")
def test_init_does_not_import_torch(tmp_path: Path) -> None:
    """Constructing GeneratorMingus must not pull torch — that only happens
    inside the subprocess."""
    import sys
    sys.modules.pop("torch", None)
    gen = GeneratorMingus(
        fork_root=_MINGUS_FORK,
        data_path=_MINGUS_DATA,
        checkpoint_dir=_MINGUS_CKPT_DIR,
        epochs=10,
        cond_pitch="D-C-B-BE-O",
        cond_duration="B-BE-O",
        device="cpu",
    )
    assert "torch" not in sys.modules, "GeneratorMingus.__init__ must not import torch"
    assert gen.fork_root == _MINGUS_FORK
    assert gen.checkpoint_dir == _MINGUS_CKPT_DIR


@pytest.mark.skipif(not _MINGUS_VENV_PY.exists(), reason="MINGUS venv not available")
@pytest.mark.skipif(not _MINGUS_DATA.is_file(), reason="MINGUS DATA.json absent")
@pytest.mark.skipif(not _MINGUS_CKPT_DIR.is_dir(), reason="MINGUS checkpoints absent")
def test_generate_end_to_end_via_subprocess(tmp_path: Path) -> None:
    gen = GeneratorMingus(
        fork_root=_MINGUS_FORK,
        data_path=_MINGUS_DATA,
        checkpoint_dir=_MINGUS_CKPT_DIR,
        epochs=10,
        cond_pitch="D-C-B-BE-O",
        cond_duration="B-BE-O",
        device="cpu",
    )
    try:
        inp = GeneratorMingusInput(
            musicxml_path=_INPUT_XML,
            seed=1, input_bars=32, output_bars=32, temperature=1.0,
        )
        out = gen.generate(inp)
        assert out.midi.instruments
        assert out.midi.instruments[0].notes
        assert isinstance(out.tempo, float)
    finally:
        gen.close()


@pytest.mark.skipif(not _MINGUS_VENV_PY.exists(), reason="MINGUS venv not available")
@pytest.mark.skipif(not _MINGUS_DATA.is_file(), reason="MINGUS DATA.json absent")
@pytest.mark.skipif(not _MINGUS_CKPT_DIR.is_dir(), reason="MINGUS checkpoints absent")
def test_generator_reuses_subprocess_across_calls(tmp_path: Path) -> None:
    """Дважды зовём gen.generate(...). Subprocess грузится один раз."""
    gen = GeneratorMingus(
        fork_root=_MINGUS_FORK,
        data_path=_MINGUS_DATA,
        checkpoint_dir=_MINGUS_CKPT_DIR,
        epochs=10,
        cond_pitch="D-C-B-BE-O",
        cond_duration="B-BE-O",
        device="cpu",
    )
    try:
        out1 = gen.generate(GeneratorMingusInput(
            musicxml_path=_INPUT_XML, seed=1, input_bars=32, output_bars=32, temperature=1.0,
        ))
        out2 = gen.generate(GeneratorMingusInput(
            musicxml_path=_INPUT_XML, seed=2, input_bars=32, output_bars=32, temperature=1.0,
        ))
        assert out1.midi.instruments and out2.midi.instruments
        assert out1.midi.instruments[0].notes != out2.midi.instruments[0].notes
    finally:
        gen.close()
