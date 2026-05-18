"""Integration smoke: each model produces all 3 output formats end-to-end.

Skipped if any model's checkpoint or venv is absent. Runs under pipeline
venv only — Generator<X> objects are constructed without importing torch.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PIPELINE_PY = _REPO_ROOT / "pipelines" / "generation-pipeline" / ".venv" / "bin" / "python"
_RUN_PY = _REPO_ROOT / "pipelines" / "generation-pipeline" / "runners" / "run.py"
_CONFIGS = _REPO_ROOT / "pipelines" / "generation-pipeline" / "runners" / "example_configs"
_INPUTS = _REPO_ROOT / "pipelines" / "generation-pipeline" / "inputs" / "musicxml"

_CMT_FORK = _REPO_ROOT / "models" / "CMT-pytorch"
_CMT_VENV_PY = _CMT_FORK / ".venv" / "bin" / "python"
_CMT_HPARAMS = _CMT_FORK / "hparams_jazz_16bars.yaml"
_CMT_CHECKPOINT = _CMT_FORK / "result" / "paper" / "16bars" / "best_jazz_model_16bars.pth.tar"


@pytest.mark.parametrize(
    "model_name,config_file,input_xml",
    [
        ("cmt",      "cmt.yaml",      "Autumn_Leaves_8bars.musicxml"),
        ("mingus",   "mingus.yaml",   "Autumn_Leaves.musicxml"),
        ("bebopnet", "bebopnet.yaml", "Autumn_Leaves.musicxml"),
    ],
)
def test_run_py_produces_three_artifacts(
    tmp_path: Path,
    model_name: str,
    config_file: str,
    input_xml: str,
) -> None:
    config = _CONFIGS / config_file
    if not config.is_file():
        pytest.skip(f"{config} missing")
    out_dir = tmp_path / model_name
    cmd = [
        str(_PIPELINE_PY), str(_RUN_PY), str(config),
        "--input", str(_INPUTS / input_xml),
        "--output-dir", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(
            f"run.py failed for {model_name}:\n"
            f"--stdout--\n{res.stdout}\n"
            f"--stderr--\n{res.stderr}"
        )

    stem = out_dir.name
    assert (out_dir / f"{stem}.mid").is_file()
    assert (out_dir / f"{stem}.musicxml").is_file()
    assert (out_dir / f"{stem}_with_chords.musicxml").is_file()


@pytest.mark.skipif(
    not _CMT_VENV_PY.exists() or not _CMT_CHECKPOINT.exists(),
    reason="CMT venv or checkpoint not available",
)
def test_cmt_subprocess_reused_across_calls(tmp_path: Path) -> None:
    """E2E: 2 генерации на одном GeneratorCmt.

    Первый вызов включает spawn + checkpoint load (>5s на CPU);
    второй — только inference. Второй должен быть значительно быстрее
    первого, что подтверждает переиспользование subprocess.
    """
    from models.cmt import GeneratorCmt, GeneratorCmtInput

    gen = GeneratorCmt(
        fork_root=_CMT_FORK,
        hparams_yaml_path=_CMT_HPARAMS,
        checkpoint_path=_CMT_CHECKPOINT,
        device="cpu",
    )
    inp = GeneratorCmtInput(
        musicxml_path=_INPUTS / "Autumn_Leaves_8bars.musicxml",
        seed=1,
        input_bars=8,
        output_bars=8,
        topk=5,
    )
    try:
        t0 = time.perf_counter()
        out1 = gen.generate(inp)
        first = time.perf_counter() - t0

        # После первого вызова subprocess должен быть живой (lazy-spawned).
        assert gen._client._proc is not None, "subprocess не был запущен"
        pid_after_first = gen._client._proc.pid
        assert gen._client._proc.poll() is None, "subprocess умер после первого вызова"

        t0 = time.perf_counter()
        out2 = gen.generate(inp)
        second = time.perf_counter() - t0

        # Ключевая проверка переиспользования: PID не изменился.
        assert gen._client._proc is not None, "subprocess исчез после второго вызова"
        pid_after_second = gen._client._proc.pid
        assert pid_after_first == pid_after_second, (
            f"subprocess был respawned: pid {pid_after_first} → {pid_after_second}"
        )

        # Оба вызова вернули валидный MIDI.
        assert out1.midi is not None
        assert out2.midi is not None

        # Информационная проверка: если второй > первого — скорее всего
        # inference стал тяжелее чем spawn, что само по себе подозрительно.
        # Не падаем, только логируем через print (виден при pytest -s).
        print(f"\n  first={first:.2f}s  second={second:.2f}s  pid={pid_after_first}")
    finally:
        gen.close()
