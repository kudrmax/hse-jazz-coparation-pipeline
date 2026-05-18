"""Pipeline-side helper to invoke a forked-venv inference runner.

Contract:
  1. Pipeline writes `request` (a JSON-serialisable dict) to a temp file.
  2. Pipeline runs `<venv_python> <runner_script> <request_path> <response_path>`.
  3. Forked-venv runner reads request JSON, does inference, writes response JSON.
  4. Pipeline reads response JSON, returns it as a dict, cleans both temp files.

The runner may also write side artifacts (e.g. midi) to disk; their target
paths are part of `request`. The helper itself only deals with the JSON IPC.

TODO: subprocess.run is invoked without a timeout. A frozen forked-venv
runner will block the pipeline indefinitely. Add `timeout_sec` parameter
once batch runs in comparation-pipeline expose realistic upper bounds.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class SubprocessInferenceError(RuntimeError):
    """Raised when a forked-venv subprocess fails or misbehaves."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_subprocess_inference(
    *,
    venv_python: Path,
    runner_script: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a forked-venv inference runner with a JSON request, return response."""
    if not venv_python.is_file():
        raise SubprocessInferenceError(
            f"venv python not found at {venv_python}; "
            "create the venv and install the model's dependencies"
        )
    with tempfile.TemporaryDirectory(prefix="pipeline_ipc_") as tmpdir:
        tmp = Path(tmpdir)
        req_path = tmp / "request.json"
        resp_path = tmp / "response.json"
        req_path.write_text(json.dumps(request))

        cmd = [str(venv_python), str(runner_script), str(req_path), str(resp_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise SubprocessInferenceError(
                f"subprocess exited with status {proc.returncode}: {runner_script.name}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        if not resp_path.is_file():
            raise SubprocessInferenceError(
                f"response file was not written by {runner_script.name}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        try:
            response: dict[str, Any] = json.loads(resp_path.read_text())
        except json.JSONDecodeError as e:
            raise SubprocessInferenceError(
                f"response file from {runner_script.name} is not valid JSON: {e}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            ) from e
        return response
