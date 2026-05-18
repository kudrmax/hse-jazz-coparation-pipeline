"""GeneratorCmt now runs torch inference via subprocess. Init no longer
loads the model; that work happens on every generate() inside the forked
venv subprocess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from models.cmt import GeneratorCmt, GeneratorCmtInput

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CMT_FORK = _REPO_ROOT / "models" / "CMT-pytorch"
_CMT_HPARAMS = _CMT_FORK / "hparams_jazz_16bars.yaml"
_CMT_CKPT = _CMT_FORK / "result" / "paper" / "16bars" / "best_jazz_model_16bars.pth.tar"
_INPUT_XML = _REPO_ROOT / "pipelines" / "generation-pipeline" / "inputs" / "musicxml" / "Autumn_Leaves_8bars.musicxml"


@pytest.mark.skipif(not _CMT_CKPT.is_file(), reason="CMT checkpoint not present")
def test_init_does_not_import_torch_or_load_checkpoint(tmp_path: Path) -> None:
    """Constructing GeneratorCmt must not pull torch / load weights — those
    only happen inside the subprocess. This is what lets pipeline venv
    (which has no torch) construct the generator."""
    import sys
    sys.modules.pop("torch", None)
    gen = GeneratorCmt(
        fork_root=_CMT_FORK,
        hparams_yaml_path=_CMT_HPARAMS,
        checkpoint_path=_CMT_CKPT,
        device="cpu",
    )
    assert "torch" not in sys.modules, "GeneratorCmt.__init__ must not import torch"
    # public attributes pipeline relies on (post-processor uses none of these,
    # but they may still be useful in __repr__ etc.):
    assert gen.fork_root == _CMT_FORK
    assert gen.checkpoint_path == _CMT_CKPT


@pytest.mark.skipif(not _CMT_CKPT.is_file(), reason="CMT checkpoint not present")
def test_generate_end_to_end_via_subprocess(tmp_path: Path) -> None:
    gen = GeneratorCmt(
        fork_root=_CMT_FORK,
        hparams_yaml_path=_CMT_HPARAMS,
        checkpoint_path=_CMT_CKPT,
        device="cpu",
    )
    inp = GeneratorCmtInput(
        musicxml_path=_INPUT_XML,
        seed=1, input_bars=8, output_bars=8, topk=5,
    )
    out = gen.generate(inp)
    assert out.midi.instruments
    assert out.midi.instruments[0].notes
    assert isinstance(out.transpose_semitones, int)


_CMT_VENV_PY = _CMT_FORK / ".venv" / "bin" / "python"


@pytest.mark.skipif(
    not _CMT_VENV_PY.exists() or not _CMT_CKPT.is_file(),
    reason="CMT venv or checkpoint not available",
)
def test_generator_reuses_subprocess_across_calls(tmp_path: Path) -> None:
    """Дважды зовём gen.generate(...). Subprocess грузится один раз."""
    gen = GeneratorCmt(
        fork_root=_CMT_FORK,
        hparams_yaml_path=_CMT_HPARAMS,
        checkpoint_path=_CMT_CKPT,
        device="cpu",
    )
    try:
        out1 = gen.generate(GeneratorCmtInput(
            musicxml_path=_INPUT_XML, seed=1, input_bars=8, output_bars=8, topk=5,
        ))
        out2 = gen.generate(GeneratorCmtInput(
            musicxml_path=_INPUT_XML, seed=2, input_bars=8, output_bars=8, topk=5,
        ))
        assert out1.midi.instruments and out2.midi.instruments
        # Different seeds → different output (статистический smoke).
        assert out1.midi.instruments[0].notes != out2.midi.instruments[0].notes
    finally:
        gen.close()
