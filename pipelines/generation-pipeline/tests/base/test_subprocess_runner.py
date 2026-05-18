"""Tests for the pipeline-side subprocess IPC helper."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from models.base.subprocess_runner import SubprocessInferenceError, run_subprocess_inference


def _write_runner(path: Path, body: str) -> None:
    """Helper: write a small Python runner script for tests to invoke."""
    path.write_text(textwrap.dedent(body))


def test_round_trips_request_and_response(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    _write_runner(runner, """
        import json, sys
        req = json.loads(open(sys.argv[1]).read())
        resp = {"echo": req["payload"], "n": req["n"] * 2}
        open(sys.argv[2], "w").write(json.dumps(resp))
    """)
    response = run_subprocess_inference(
        venv_python=Path(sys.executable),  # use current python as a stand-in
        runner_script=runner,
        request={"payload": "hello", "n": 21},
    )
    assert response == {"echo": "hello", "n": 42}


def test_raises_with_stderr_on_nonzero_exit(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    _write_runner(runner, """
        import sys
        sys.stderr.write("kaboom\\n")
        sys.exit(7)
    """)
    with pytest.raises(SubprocessInferenceError) as exc:
        run_subprocess_inference(
            venv_python=Path(sys.executable),
            runner_script=runner,
            request={},
        )
    assert exc.value.returncode == 7
    assert "kaboom" in exc.value.stderr


def test_missing_response_raises(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    # Runner exits 0 but never writes the response file.
    _write_runner(runner, "import sys\nsys.exit(0)\n")
    with pytest.raises(SubprocessInferenceError, match="response file was not written"):
        run_subprocess_inference(
            venv_python=Path(sys.executable),
            runner_script=runner,
            request={},
        )


def test_venv_python_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(SubprocessInferenceError, match="venv python not found"):
        run_subprocess_inference(
            venv_python=tmp_path / "does-not-exist" / "python",
            runner_script=tmp_path / "runner.py",
            request={},
        )


def test_malformed_response_raises(tmp_path: Path) -> None:
    """Runner exits 0 but writes invalid JSON — should raise SubprocessInferenceError,
    not bare json.JSONDecodeError, so callers have a single error type to handle."""
    runner = tmp_path / "runner.py"
    _write_runner(runner, """
        import sys
        open(sys.argv[2], "w").write("not json at all")
        sys.exit(0)
    """)
    with pytest.raises(SubprocessInferenceError, match="not valid JSON"):
        run_subprocess_inference(
            venv_python=Path(sys.executable),
            runner_script=runner,
            request={},
        )
