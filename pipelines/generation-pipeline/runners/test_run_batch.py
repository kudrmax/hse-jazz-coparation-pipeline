"""Tests for run_batch — единая функция, под которой работают и single-input
и --batch режимы run.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
RUNNERS_DIR = PIPELINE_ROOT / "runners"
# RUNNERS_DIR must be first in sys.path so `import run` finds runners/run.py,
# not models/CMT-pytorch/run.py. Insert PIPELINE_ROOT first, then RUNNERS_DIR
# (each insert(0) prepends → last insert ends up at index 0).
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(RUNNERS_DIR))

CMT_VENV_PY = REPO_ROOT / "models/CMT-pytorch/.venv/bin/python"


@pytest.mark.skipif(not CMT_VENV_PY.exists(), reason="CMT venv not available")
def test_run_batch_single_task_writes_outputs(tmp_path):
    """Один task — пишет MIDI согласно output.formats."""
    from run import run_batch
    from run_config import (
        CmtConfig, CommonConfig, OutputConfig, RunConfig,
    )

    cfg = RunConfig(
        model="cmt",
        output=OutputConfig(formats=("midi",), force_overwrite=True),
        common=CommonConfig(seed=1, input_bars=8, output_bars=8, device="cpu"),
        model_params=CmtConfig(
            fork_root=REPO_ROOT / "models/CMT-pytorch",
            hparams_yaml_path=REPO_ROOT / "models/CMT-pytorch/hparams_jazz_16bars.yaml",
            checkpoint_path=REPO_ROOT / "models/CMT-pytorch/result/paper/16bars/best_jazz_model_16bars.pth.tar",
            topk=5,
        ),
    )
    tasks = [{
        "task_id": "single",
        "input": str(REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"),
        "output_dir": str(tmp_path / "single"),
        "output_stem": "single",
    }]
    results_path = tmp_path / "results.jsonl"
    run_batch(cfg, tasks, results_path=results_path)

    midi = tmp_path / "single" / "single.mid"
    assert midi.is_file()
    lines = results_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["ok"] is True
    assert str(midi) in rec["files"]
