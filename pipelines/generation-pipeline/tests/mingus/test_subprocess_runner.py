"""Tests for MINGUS subprocess runner — server-mode protocol."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "pipelines/generation-pipeline/models/mingus/_subprocess_runner.py"
MINGUS_VENV_PY = REPO_ROOT / "models/MINGUS/.venv/bin/python"
MINGUS_FORK = REPO_ROOT / "models/MINGUS"
DATA_PATH = MINGUS_FORK / "A_preprocessData/data/DATA.json"
CKPT_DIR = MINGUS_FORK / "B_train/models/paper-optimal"
THEME_XML = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves.musicxml"


@pytest.mark.skipif(not MINGUS_VENV_PY.exists(), reason="MINGUS venv not available")
def test_server_mode_two_consecutive_requests(tmp_path):
    """Запускаем runner в server-mode, шлём 2 запроса подряд через stdin."""
    midi_out_1 = tmp_path / "out_1.mid"
    midi_out_2 = tmp_path / "out_2.mid"
    config = {
        "fork_root": str(MINGUS_FORK),
        "data_path": str(DATA_PATH),
        "checkpoint_dir": str(CKPT_DIR),
        "epochs": 10,
        "cond_pitch": "D-C-B-BE-O",
        "cond_duration": "B-BE-O",
        "device": "cpu",
    }
    requests = [
        {"musicxml_path": str(THEME_XML), "seed": 1, "input_bars": 32,
         "output_bars": 32, "temperature": 1.0, "midi_out_path": str(midi_out_1)},
        {"musicxml_path": str(THEME_XML), "seed": 2, "input_bars": 32,
         "output_bars": 32, "temperature": 1.0, "midi_out_path": str(midi_out_2)},
    ]
    proc = subprocess.Popen(
        [str(MINGUS_VENV_PY), str(RUNNER), "--server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        proc.stdin.write(json.dumps(config) + "\n")
        proc.stdin.flush()
        responses = []
        for req in requests:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, f"no response; stderr: {proc.stderr.read()}"
            responses.append(json.loads(line))
        proc.stdin.close()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0
    for resp in responses:
        assert resp["ok"] is True
        assert "tempo" in resp
        assert "title" in resp
    assert midi_out_1.is_file() and midi_out_2.is_file()
