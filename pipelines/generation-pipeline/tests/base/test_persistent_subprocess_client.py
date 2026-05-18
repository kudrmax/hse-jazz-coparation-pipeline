"""Tests for PersistentSubprocessClient — pipeline-side wrapper over a
long-lived forked subprocess that speaks JSON-line over stdin/stdout."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from models.base.persistent_subprocess_client import (
    PersistentSubprocessClient,
    SubprocessInferenceError,
)


def _write_server_runner(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_round_trip_single_request(tmp_path: Path) -> None:
    """One request → one response, subprocess stays alive."""
    runner = tmp_path / "runner.py"
    _write_server_runner(runner, """
        import json, sys
        # No setup config — read it but ignore.
        sys.stdin.readline()
        for line in sys.stdin:
            req = json.loads(line)
            resp = {"echo": req["payload"], "n": req["n"] * 2}
            print(json.dumps(resp), flush=True)
    """)
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={"dummy": "cfg"},
    )
    try:
        resp = client.request({"payload": "hello", "n": 21})
        assert resp == {"echo": "hello", "n": 42}
    finally:
        client.close()


def test_handles_stderr_chatter_without_deadlock(tmp_path: Path) -> None:
    """If the subprocess writes lots to stderr before responding, the client
    should NOT deadlock. Regression for the stderr=PIPE-no-drain bug.
    Without a drain thread this test would hang until pytest timeout."""
    runner = tmp_path / "runner.py"
    _write_server_runner(runner, """
        import json, sys
        sys.stdin.readline()  # setup config
        for line in sys.stdin:
            req = json.loads(line)
            # Write 200 KB of stderr noise — больше чем pipe buffer (~64KB).
            for _ in range(2000):
                sys.stderr.write("x" * 100 + "\\n")
            sys.stderr.flush()
            print(json.dumps({"ok": True, "n": req["n"]}), flush=True)
    """)
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={},
    )
    try:
        for i in range(3):
            resp = client.request({"n": i})
            assert resp == {"ok": True, "n": i}
    finally:
        client.close()


def test_setup_runs_once_across_multiple_requests(tmp_path: Path) -> None:
    """Setup line читается один раз, дальше — только request'ы.
    Симулируем дорогую загрузку через счётчик в файле."""
    counter_file = tmp_path / "load_count.txt"
    counter_file.write_text("0")
    runner = tmp_path / "runner.py"
    _write_server_runner(runner, f"""
        import json, sys
        from pathlib import Path
        cf = Path({str(counter_file)!r})
        # Setup line: имитируем загрузку. Должна выполниться один раз.
        sys.stdin.readline()
        cf.write_text(str(int(cf.read_text()) + 1))
        for line in sys.stdin:
            req = json.loads(line)
            print(json.dumps({{"id": req["id"]}}), flush=True)
    """)
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={"setup": "once"},
    )
    try:
        for i in range(5):
            resp = client.request({"id": i})
            assert resp == {"id": i}
    finally:
        client.close()
    assert counter_file.read_text() == "1", "setup должен выполняться только один раз"


def test_close_is_idempotent(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    _write_server_runner(runner, """
        import sys
        sys.stdin.readline()
        for line in sys.stdin:
            print('{"ok": true}', flush=True)
    """)
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={},
    )
    client.request({"x": 1})
    client.close()
    client.close()  # должно быть noop


def test_close_without_request_is_noop(tmp_path: Path) -> None:
    """Если client.request() ни разу не звался — close() ничего не делает."""
    runner = tmp_path / "nonexistent.py"  # spawn даже не пытались
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={},
    )
    client.close()  # не должен бросить


def test_request_after_subprocess_died_raises(tmp_path: Path) -> None:
    """Если subprocess умер посреди жизни — request() поднимает SubprocessInferenceError."""
    runner = tmp_path / "runner.py"
    _write_server_runner(runner, """
        import json, sys
        sys.stdin.readline()
        line = sys.stdin.readline()
        # Делаем один ответ и умираем.
        print(json.dumps({"ok": True}), flush=True)
        sys.exit(13)
    """)
    client = PersistentSubprocessClient(
        venv_python=Path(sys.executable),
        runner_script=runner,
        config={},
    )
    try:
        first = client.request({"x": 1})
        assert first == {"ok": True}
        with pytest.raises(SubprocessInferenceError):
            client.request({"x": 2})
    finally:
        client.close()


def test_missing_venv_python_raises_at_init(tmp_path: Path) -> None:
    with pytest.raises(SubprocessInferenceError, match="venv python not found"):
        PersistentSubprocessClient(
            venv_python=tmp_path / "no-such-python",
            runner_script=tmp_path / "runner.py",
            config={},
        )
