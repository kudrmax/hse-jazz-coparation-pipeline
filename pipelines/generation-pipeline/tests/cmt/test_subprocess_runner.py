"""Tests for CMT subprocess runner — server-mode protocol."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "pipelines/generation-pipeline/models/cmt/_subprocess_runner.py"
CMT_VENV_PY = REPO_ROOT / "models/CMT-pytorch/.venv/bin/python"
CMT_HPARAMS = REPO_ROOT / "models/CMT-pytorch/hparams_jazz_16bars.yaml"
CMT_CKPT = REPO_ROOT / "models/CMT-pytorch/result/paper/16bars/best_jazz_model_16bars.pth.tar"
THEME_XML = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"


@pytest.mark.skipif(not CMT_VENV_PY.exists(), reason="CMT venv not available")
def test_server_mode_two_consecutive_requests(tmp_path):
    """Запускаем runner в server-mode, шлём 2 запроса подряд через stdin."""
    midi_out_1 = tmp_path / "out_1.mid"
    midi_out_2 = tmp_path / "out_2.mid"
    config = {
        "fork_root": str(REPO_ROOT / "models/CMT-pytorch"),
        "hparams_yaml_path": str(CMT_HPARAMS),
        "checkpoint_path": str(CMT_CKPT),
        "device": "cpu",
    }
    requests = [
        {"musicxml_path": str(THEME_XML), "seed": 1, "input_bars": 8,
         "output_bars": 8, "topk": 5, "midi_out_path": str(midi_out_1)},
        {"musicxml_path": str(THEME_XML), "seed": 2, "input_bars": 8,
         "output_bars": 8, "topk": 5, "midi_out_path": str(midi_out_2)},
    ]
    proc = subprocess.Popen(
        [str(CMT_VENV_PY), str(RUNNER), "--server"],
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
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0
    for resp in responses:
        assert resp["ok"] is True
        assert "transpose_semitones" in resp
    assert midi_out_1.is_file() and midi_out_2.is_file()
