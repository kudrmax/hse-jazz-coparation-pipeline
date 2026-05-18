"""Tests for MetricsPhase + PerSegmentMetricsRunner."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from comparation_config import (
    BebopnetModelConfig, CmtModelConfig, ComparationConfig,
    MingusModelConfig, SegmentationConfig,
)
from manifest import Manifest
from phases.metrics import MetricsPhase, MetricsPhaseError, PerSegmentMetricsRunner


def _stub_cfg() -> ComparationConfig:
    return ComparationConfig(
        slug="t", samples_per_theme=1, device="cpu",
        themes_limit="all", output_formats=("midi",),
        segmentation=SegmentationConfig(chunk_bars=8),
        cmt=CmtModelConfig(
            fork_root=Path("/cmt"), hparams_yaml_path=Path("/hp"),
            checkpoint_path=Path("/ck"), topk=5,
            input_bars=8, output_bars=8,
        ),
        mingus=MingusModelConfig(
            fork_root=Path("/m"), data_path=Path("/m/d"),
            checkpoint_dir=Path("/m/c"), epochs=10,
            cond_pitch="x", cond_duration="y",
            temperature=1.0, input_bars=8, output_bars=8,
        ),
        bebopnet=BebopnetModelConfig(
            fork_root=Path("/b"), model_dir=Path("/m"),
            checkpoint="c", temperature=1.0,
            input_bars=8, output_bars=8,
        ),
    )


def test_metrics_phase_missing_manifest_raises(tmp_path):
    cfg = _stub_cfg()
    with pytest.raises(MetricsPhaseError):
        MetricsPhase(
            per_segment_runner=PerSegmentMetricsRunner(metrics=[]),
            corpus_pipelines=[],
        ).run(tmp_path / "missing", cfg)


def test_metrics_phase_calls_runner_pipelines_then_final_table(tmp_path, monkeypatch):
    """run() вызывает per_segment_runner, потом каждый corpus_pipeline, потом FinalTableStep."""
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("t", "sha256:abc", 1, ("midi",))
    m.save_atomic()

    calls = []

    class _FakeRunner:
        def run(self, slug_dir, manifest, chunk_bars):
            calls.append(("runner", chunk_bars))

    class _FakePipeline:
        def __init__(self, name):
            self.name = name

        def run(self, slug_dir, manifest, cfg):
            calls.append(("pipeline", self.name))

    from phases import metrics as metrics_mod

    class _FakeFinalTableStep:
        def __init__(self, slug_dir):
            self.slug_dir = slug_dir

        def run(self):
            calls.append(("final_table", self.slug_dir.name))

    monkeypatch.setattr(metrics_mod, "FinalTableStep", _FakeFinalTableStep)

    MetricsPhase(
        per_segment_runner=_FakeRunner(),
        corpus_pipelines=[_FakePipeline("a"), _FakePipeline("b")],
    ).run(tmp_path, _stub_cfg())

    assert calls == [
        ("runner", 8),
        ("pipeline", "a"),
        ("pipeline", "b"),
        ("final_table", tmp_path.name),
    ]


def test_per_segment_runner_empty_writes_header_only(tmp_path):
    """Если нет sample-dir'ов на диске — per_segment.csv только с header'ом."""
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("t", "sha256:abc", 1, ("midi",))
    # без themes — active_themes() пустой

    PerSegmentMetricsRunner(metrics=[]).run(tmp_path, m, chunk_bars=8)

    per_seg = tmp_path / "_metrics" / "per_segment.csv"
    assert per_seg.exists()
    # Только header (1 строка)
    assert len(per_seg.read_text().splitlines()) == 1
