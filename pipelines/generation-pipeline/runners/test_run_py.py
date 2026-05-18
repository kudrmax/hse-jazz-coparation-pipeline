"""Unit tests for run.py auto-bars resolution and batch CLI.

Run from pipeline venv:
    pipelines/generation-pipeline/.venv/bin/python -m pytest \\
        pipelines/generation-pipeline/runners/test_run_py.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNNERS_DIR = Path(__file__).resolve().parent
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_resolve_auto_bars_replaces_auto_with_count() -> None:
    """auto-значения подменяются на count_bars_from_xml(--input)."""
    import run
    from run_config import CommonConfig

    common_in = CommonConfig(
        seed=1, input_bars="auto", output_bars="auto", device="cpu",
    )
    input_path = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"
    common_out = run._resolve_auto_bars(common_in, input_path)
    assert common_out.input_bars == 8
    assert common_out.output_bars == 8


def test_resolve_auto_bars_leaves_int_alone() -> None:
    """Числовые значения не трогаются."""
    import run
    from run_config import CommonConfig

    common_in = CommonConfig(
        seed=42, input_bars=8, output_bars=16, device="cpu",
    )
    input_path = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"
    common_out = run._resolve_auto_bars(common_in, input_path)
    assert common_out.input_bars == 8
    assert common_out.output_bars == 16
    assert common_out.seed == 42


def test_resolve_auto_bars_mixed() -> None:
    """Смешанный случай: input_bars=auto, output_bars=int."""
    import run
    from run_config import CommonConfig

    common_in = CommonConfig(
        seed=1, input_bars="auto", output_bars=16, device="cpu",
    )
    input_path = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"
    common_out = run._resolve_auto_bars(common_in, input_path)
    assert common_out.input_bars == 8
    assert common_out.output_bars == 16


import pytest  # noqa: E402


@pytest.mark.skipif(
    not (REPO_ROOT / "models/CMT-pytorch/.venv/bin/python").exists(),
    reason="CMT venv not available"
)
def test_cli_batch_mode_runs_all_tasks(tmp_path: Path) -> None:
    """run.py --batch tasks.jsonl --output results.jsonl создаёт все mid + results."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        (REPO_ROOT / "pipelines/generation-pipeline/runners/example_configs/cmt.yaml").read_text()
    )
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("\n".join([
        json.dumps({
            "task_id": f"t{i}",
            "input": str(REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"),
            "output_dir": str(tmp_path / f"t{i}"),
            "output_stem": "out",
        })
        for i in range(2)
    ]))
    results_path = tmp_path / "results.jsonl"
    proc = subprocess.run(
        [str(REPO_ROOT / "pipelines/generation-pipeline/.venv/bin/python"),
         str(REPO_ROOT / "pipelines/generation-pipeline/runners/run.py"),
         str(cfg_path),
         "--batch", str(tasks_path),
         "--output", str(results_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert results_path.is_file()
    lines = results_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["ok"] for line in lines)
