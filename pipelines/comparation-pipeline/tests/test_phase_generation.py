"""Tests for GenerationPhase."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from comparation_config import (
    BebopnetModelConfig, CmtModelConfig, ComparationConfig,
    MingusModelConfig, SegmentationConfig,
)
from manifest import Manifest
from orchestrators.base import BaseModelOrchestrator
from phases.generation import GenerationPhase, GenerationPhaseError


class _RecordingOrch(BaseModelOrchestrator):
    """Mock-orchestrator: фиксирует вызов run_batch без побочных эффектов."""
    def __init__(self, name: str):
        self.name = name
        self.called = False

    def _build_tasks(self, cfg, manifest, output_dir, corpus):
        return []

    def _post_process(self, manifest, output_dir, results, returncode, cfg):
        pass

    def _derived_yaml_block(self, cfg):
        return {}

    def sample_complete(self, sample_dir, n_chunks):
        return False

    def run_batch(self, cfg, manifest, output_dir, corpus):
        self.called = True


def _write_minimal_yaml(path: Path, slug: str) -> None:
    """Минимальный config.yaml, который пройдёт load_config + fingerprint."""
    payload = {
        "slug": slug, "samples_per_theme": 1, "device": "cpu",
        "themes_limit": "all", "output_formats": ["midi"],
        "segmentation": {"chunk_bars": 8},
        "cmt": {
            "fork_root": "models/CMT-pytorch",
            "hparams_yaml_path": "models/CMT-pytorch/hparams_jazz_16bars.yaml",
            "checkpoint_path": "models/CMT-pytorch/result/paper/16bars/best_jazz_model_16bars.pth.tar",
            "topk": 5, "input_bars": 8, "output_bars": 8,
        },
        "mingus": {
            "fork_root": "models/MINGUS",
            "data_path": "models/MINGUS/A_preprocessData/data/DATA.json",
            "checkpoint_dir": "models/MINGUS/B_train/models/paper-optimal",
            "epochs": 10,
            "cond_pitch": "D-C-B-BE-O", "cond_duration": "B-BE-O",
            "temperature": 1.0,
            "input_bars": "auto", "output_bars": "auto",
        },
        "bebopnet": {
            "fork_root": "models/bebopnet-code",
            "model_dir": "models/bebopnet-code/result/paper-default",
            "checkpoint": "model_best.pt",
            "temperature": 1.0,
            "input_bars": "auto", "output_bars": "auto",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _stub_cfg(slug: str = "t") -> ComparationConfig:
    return ComparationConfig(
        slug=slug, samples_per_theme=1, device="cpu",
        themes_limit="all", output_formats=("midi",),
        segmentation=SegmentationConfig(chunk_bars=8),
        cmt=CmtModelConfig(
            fork_root=Path("/cmt"), hparams_yaml_path=Path("/hp"),
            checkpoint_path=Path("/ck"), topk=5,
            input_bars=8, output_bars=8,
        ),
        mingus=MingusModelConfig(
            fork_root=Path("/m"), data_path=Path("/m/data"),
            checkpoint_dir=Path("/m/ck"), epochs=10,
            cond_pitch="D-C-B-BE-O", cond_duration="B-BE-O",
            temperature=1.0, input_bars=8, output_bars=8,
        ),
        bebopnet=BebopnetModelConfig(
            fork_root=Path("/b"), model_dir=Path("/md"),
            checkpoint="ck.pt", temperature=1.0,
            input_bars=8, output_bars=8,
        ),
    )


def test_generation_phase_calls_each_orchestrator(tmp_path, monkeypatch):
    """run() вызывает run_batch на каждом orchestrator'е в порядке."""
    # Подменим SelfHealer на no-op, чтобы не лезть в corpus
    import phases.generation as gen_mod

    class _NoopHealer:
        def __init__(self, orchestrators):
            pass

        def sync(self, *a, **kw):
            pass

    monkeypatch.setattr(gen_mod, "SelfHealer", _NoopHealer)

    yaml_path = tmp_path / "t.yaml"
    _write_minimal_yaml(yaml_path, "t")

    output_dir = tmp_path / "outputs" / "t"

    o1 = _RecordingOrch("cmt")
    o2 = _RecordingOrch("mingus")
    o3 = _RecordingOrch("bebopnet")

    # cfg достаточен для bootstrap'а manifest'а
    cfg = _stub_cfg("t")

    # Минуем настоящий load_config, чтобы не валидировать пути в YAML
    phase = GenerationPhase([o1, o2, o3])
    phase.run(cfg, yaml_path, output_dir)

    assert o1.called and o2.called and o3.called


def test_generation_phase_fingerprint_mismatch_raises(tmp_path, monkeypatch):
    """Если manifest.config_fingerprint != current — GenerationPhaseError."""
    import phases.generation as gen_mod

    class _NoopHealer:
        def __init__(self, orchestrators):
            pass

        def sync(self, *a, **kw):
            pass

    monkeypatch.setattr(gen_mod, "SelfHealer", _NoopHealer)

    yaml_path = tmp_path / "t.yaml"
    _write_minimal_yaml(yaml_path, "t")

    output_dir = tmp_path / "outputs" / "t"
    output_dir.mkdir(parents=True)

    # подсунем manifest с устаревшим fingerprint'ом
    stale = Manifest(path=output_dir / "manifest.json")
    stale.bootstrap("t", "sha256:STALE_FINGERPRINT", 1, ("midi",))
    stale.save_atomic()

    cfg = _stub_cfg("t")
    phase = GenerationPhase([])

    with pytest.raises(GenerationPhaseError):
        phase.run(cfg, yaml_path, output_dir)


def test_generation_phase_writes_failures_txt(tmp_path, monkeypatch):
    """_write_failures_txt создаёт пустой файл когда нет failed pairs."""
    import phases.generation as gen_mod

    class _NoopHealer:
        def __init__(self, orchestrators):
            pass

        def sync(self, *a, **kw):
            pass

    monkeypatch.setattr(gen_mod, "SelfHealer", _NoopHealer)

    yaml_path = tmp_path / "t.yaml"
    _write_minimal_yaml(yaml_path, "t")
    output_dir = tmp_path / "outputs" / "t"
    cfg = _stub_cfg("t")
    phase = GenerationPhase([])
    phase.run(cfg, yaml_path, output_dir)
    assert (output_dir / "_failures.txt").exists()
    assert (output_dir / "_failures.txt").read_text() == ""
