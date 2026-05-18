"""Smoke for the BebopNet vocab-dump runner."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from models.base.subprocess_runner import run_subprocess_inference

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BBN_VENV_PY = _REPO_ROOT / "models" / "bebopnet-code" / ".venv" / "bin" / "python"
_BBN_RUNNER = _REPO_ROOT / "pipelines" / "generation-pipeline" / "models" / "bebopnet" / "_vocab_dump_runner.py"
_BBN_FORK = _REPO_ROOT / "models" / "bebopnet-code"
_BBN_MODEL_DIR = _BBN_FORK / "result" / "paper-default"


@pytest.mark.skipif(not _BBN_VENV_PY.is_file(), reason="bebopnet venv not built")
@pytest.mark.skipif(not (_BBN_MODEL_DIR / "converter_and_duration.pkl").is_file(),
                    reason="bebopnet converter pickle missing")
def test_vocab_dump_returns_durations(tmp_path: Path) -> None:
    request = {
        "fork_root": str(_BBN_FORK),
        "model_dir": str(_BBN_MODEL_DIR),
    }
    response = run_subprocess_inference(
        venv_python=_BBN_VENV_PY,
        runner_script=_BBN_RUNNER,
        request=request,
    )
    durations_raw = response["durations"]
    assert isinstance(durations_raw, list)
    assert len(durations_raw) > 0
    durations = {Fraction(num, den) for num, den in durations_raw}
    # The vocabulary must include common note durations.
    assert Fraction(1, 1) in durations  # quarter note
    assert Fraction(1, 2) in durations  # eighth
