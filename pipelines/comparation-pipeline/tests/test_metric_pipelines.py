"""Tests for metric_pipelines registry + base ABC contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from metric_pipelines.base import BaseCorpusMetricPipeline
from metric_pipelines.bar_rhythm_jsd import BarRhythmJsdPipeline
from metric_pipelines.mgeval import MgevalPipeline
from metric_pipelines.plagiarism import PlagiarismPipeline
from metric_pipelines.registry import all_corpus_pipelines


def test_base_pipeline_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseCorpusMetricPipeline()


def test_all_corpus_pipelines_order():
    pipes = all_corpus_pipelines()
    assert [p.name for p in pipes] == ["mgeval", "bar_rhythm_jsd", "plagiarism"]
    assert isinstance(pipes[0], MgevalPipeline)
    assert isinstance(pipes[1], BarRhythmJsdPipeline)
    assert isinstance(pipes[2], PlagiarismPipeline)


def test_pipeline_csv_filenames_match_name():
    """Каждый CSV должен соответствовать name."""
    assert MgevalPipeline().csv_filename == "mgeval.csv"
    assert BarRhythmJsdPipeline().csv_filename == "bar_rhythm_jsd.csv"
    assert PlagiarismPipeline().csv_filename == "plagiarism.csv"


def test_template_method_run_call_order(tmp_path):
    """Run() вызывает hooks в правильном порядке."""
    calls = []

    class _Fake(BaseCorpusMetricPipeline):
        name = "fake"
        csv_filename = "fake.csv"

        def _load_real_corpus(self, cfg):
            calls.append("real")
            return [1, 2, 3]

        def _load_gen_corpus(self, slug_dir, model, samples_per_theme, active_themes):
            calls.append(f"gen:{model}")
            return [10, 20]

        def _compute(self, real, gen_by_model):
            calls.append("compute")
            return [{"a": 1}]

        def _write_csv(self, rows, out_path):
            calls.append(f"write:{out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("ok")

    from comparation_config import (
        BebopnetModelConfig, CmtModelConfig, ComparationConfig,
        MingusModelConfig, SegmentationConfig,
    )
    from manifest import Manifest

    cfg = ComparationConfig(
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
    m = Manifest(path=tmp_path / "m.json", samples_per_theme=1)
    _Fake().run(tmp_path, m, cfg)

    assert calls[0] == "real"
    assert "gen:cmt" in calls
    assert "gen:mingus" in calls
    assert "gen:bebopnet" in calls
    assert calls[-2] == "compute"
    assert calls[-1].startswith("write:")
