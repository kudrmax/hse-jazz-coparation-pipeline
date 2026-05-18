"""Tests for BebopNet subprocess runner — server-mode protocol."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import music21 as m21
import pytest

from models.base.subprocess_runner import run_subprocess_inference
from models.bebopnet.preprocessor import BebopnetPreprocessor

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "pipelines/generation-pipeline/models/bebopnet/_subprocess_runner.py"
BBN_VENV_PY = REPO_ROOT / "models/bebopnet-code/.venv/bin/python"
BBN_FORK = REPO_ROOT / "models/bebopnet-code"
BBN_MODEL_DIR = BBN_FORK / "result/paper-default"
BBN_VOCAB_DUMP = REPO_ROOT / "pipelines/generation-pipeline/models/bebopnet/_vocab_dump_runner.py"
THEME_XML = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves.musicxml"


def _preprocess_theme(tmp_path: Path) -> Path:
    """Препроцессим тему через BebopnetPreprocessor (vocab-safe durations)."""
    vocab_resp = run_subprocess_inference(
        venv_python=BBN_VENV_PY,
        runner_script=BBN_VOCAB_DUMP,
        request={"fork_root": str(BBN_FORK), "model_dir": str(BBN_MODEL_DIR)},
    )
    vocab = frozenset(Fraction(num, den) for num, den in vocab_resp["durations"])
    preproc = BebopnetPreprocessor(vocab=vocab)

    parsed = m21.converter.parse(str(THEME_XML))

    class _StubInput:
        def __init__(self) -> None:
            self.musicxml_path = THEME_XML
            self.input_bars = 32

        def get_musicxml_path(self) -> Path:
            return self.musicxml_path

    _, processed_path = preproc.process(_StubInput(), parsed)
    out = tmp_path / "preprocessed.musicxml"
    out.write_bytes(processed_path.read_bytes())
    if processed_path != THEME_XML:
        processed_path.unlink(missing_ok=True)
    return out


@pytest.mark.skipif(not BBN_VENV_PY.exists(), reason="BebopNet venv not available")
def test_server_mode_two_consecutive_requests(tmp_path):
    """Запускаем runner в server-mode, шлём 2 запроса подряд через stdin."""
    theme_xml = _preprocess_theme(tmp_path)
    midi_out_1 = tmp_path / "out_1.mid"
    midi_out_2 = tmp_path / "out_2.mid"
    config = {
        "fork_root": str(BBN_FORK),
        "model_dir": str(BBN_MODEL_DIR),
        "checkpoint": "model_best.pt",
        "device": "cpu",
    }
    requests = [
        {"musicxml_path": str(theme_xml), "seed": 1, "output_bars": 32,
         "temperature": 1.0, "midi_out_path": str(midi_out_1)},
        {"musicxml_path": str(theme_xml), "seed": 2, "output_bars": 32,
         "temperature": 1.0, "midi_out_path": str(midi_out_2)},
    ]
    proc = subprocess.Popen(
        [str(BBN_VENV_PY), str(RUNNER), "--server"],
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
        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0
    for resp in responses:
        assert resp["ok"] is True, f"inference failed: {resp.get('error')}\n{resp.get('traceback', '')}"
        assert "top_likelihood" in resp
    assert midi_out_1.is_file() and midi_out_2.is_file()
