"""Long-lived forked subprocess client.

Spawns the runner subprocess lazily on first request(), writes the setup
config as the first stdin line, then exchanges request/response JSON-line
pairs until close() closes stdin and the subprocess exits gracefully.

A daemon drain thread continuously reads from the subprocess's stderr into
an in-memory buffer. This prevents the well-known subprocess.PIPE deadlock
where a full stderr buffer blocks the child's writes, in turn blocking our
reads from stdout.

Replaces the older one-shot run_subprocess_inference helper.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


class SubprocessInferenceError(RuntimeError):
    """Raised when the forked-venv subprocess fails or misbehaves."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class PersistentSubprocessClient:
    def __init__(
        self,
        *,
        venv_python: Path,
        runner_script: Path,
        config: dict[str, Any],
    ) -> None:
        if not venv_python.is_file():
            raise SubprocessInferenceError(
                f"venv python not found at {venv_python}; create the venv "
                "and install the model's dependencies"
            )
        self._venv_python = venv_python
        self._runner_script = runner_script
        self._config = config
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def _spawn(self) -> None:
        """Lazy spawn on first request()."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            [str(self._venv_python), str(self._runner_script), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        # Background thread: read stderr line-by-line, append to buffer.
        # Daemon so it doesn't block interpreter shutdown if close() never called.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr_loop, daemon=True,
        )
        self._stderr_thread.start()
        # Send the setup config as the first stdin line.
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(self._config) + "\n")
        self._proc.stdin.flush()

    def _drain_stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            for line in self._proc.stderr:
                self._stderr_lines.append(line)
        except Exception:
            pass  # subprocess closed stderr; iter exits

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            self._spawn()
        assert self._proc is not None
        if self._proc.stdin is None or self._proc.stdout is None:
            raise SubprocessInferenceError(
                "subprocess pipes unavailable",
                returncode=self._proc.returncode,
                stderr=self._stderr_snapshot(),
            )

        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise SubprocessInferenceError(
                f"subprocess stdin broken — process likely died: {e}",
                returncode=self._proc.returncode,
                stderr=self._stderr_snapshot(),
            ) from e

        line = self._proc.stdout.readline()
        if not line:
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise SubprocessInferenceError(
                "subprocess produced no response (EOF) — process likely died",
                returncode=self._proc.returncode,
                stderr=self._stderr_snapshot(),
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise SubprocessInferenceError(
                f"subprocess returned non-JSON line: {line!r}: {e}",
                returncode=self._proc.returncode,
                stderr=self._stderr_snapshot(),
            ) from e

    def _stderr_snapshot(self) -> str:
        """Snapshot of stderr accumulated so far. Non-blocking."""
        return "".join(self._stderr_lines)

    def close(self) -> None:
        """Close stdin, wait for graceful exit, kill on timeout. Idempotent."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            # Close remaining fds (subprocess.Popen doesn't auto-close them).
            for fd in (self._proc.stdout, self._proc.stderr):
                if fd is not None:
                    try:
                        fd.close()
                    except Exception:
                        pass
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
        finally:
            self._proc = None
            self._stderr_thread = None
