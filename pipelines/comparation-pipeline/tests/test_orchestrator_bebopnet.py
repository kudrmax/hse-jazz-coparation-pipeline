"""Tests for BebopnetOrchestrator."""
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
from orchestrators.bebopnet import BebopnetOrchestrator


def _stub_cfg(bebopnet_input_bars=8, bebopnet_output_bars=8) -> ComparationConfig:
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
            fork_root=Path("/m"), data_path=Path("/m/data"),
            checkpoint_dir=Path("/m/ck"), epochs=10,
            cond_pitch="D-C-B-BE-O", cond_duration="B-BE-O",
            temperature=1.0, input_bars=8, output_bars=8,
        ),
        bebopnet=BebopnetModelConfig(
            fork_root=Path("/b"), model_dir=Path("/b/md"),
            checkpoint="ck.pt", temperature=0.85,
            input_bars=bebopnet_input_bars, output_bars=bebopnet_output_bars,
        ),
    )


def _make_manifest(tmp_path, themes: list[str], samples_per_theme: int = 1) -> Manifest:
    m = Manifest(path=tmp_path / "manifest.json", samples_per_theme=samples_per_theme)
    for t in themes:
        m.add_theme(t)
    return m


def test_bebopnet_name():
    assert BebopnetOrchestrator().name == "bebopnet"


def test_bebopnet_derived_yaml_block():
    cfg = _stub_cfg()
    b = BebopnetOrchestrator()._derived_yaml_block(cfg)
    assert b["model"] == "bebopnet"
    assert b["common"]["input_bars"] == 8
    assert b["bebopnet"]["model_dir"] == "/b/md"
    assert b["bebopnet"]["checkpoint"] == "ck.pt"
    assert b["bebopnet"]["temperature"] == 0.85


def test_bebopnet_sample_complete(tmp_path):
    sd = tmp_path / "s"
    sd.mkdir()
    assert BebopnetOrchestrator().sample_complete(sd, n_chunks=3) is False
    (sd / "raw_full.mid").write_bytes(b"x")
    assert BebopnetOrchestrator().sample_complete(sd, n_chunks=3) is True


def test_bebopnet_build_tasks_skips_if_raw_full_exists(tmp_path):
    theme = "T"
    out = tmp_path / "out"
    chunks_dir = out / "themes" / theme / "theme_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_0.musicxml").write_text("<x/>")
    sample_dir = out / "themes" / theme / "bebopnet" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "raw_full.mid").write_bytes(b"x")
    m = _make_manifest(tmp_path, [theme])
    cfg = _stub_cfg()
    assert BebopnetOrchestrator()._build_tasks(cfg, m, out, corpus=None) == []


def test_bebopnet_build_tasks_auto_bars_calls_corpus_count_bars(tmp_path):
    theme = "T"
    out = tmp_path / "out"
    chunks_dir = out / "themes" / theme / "theme_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_0.musicxml").write_text("<x/>")
    (out / "themes" / theme / "theme.musicxml").write_text("<x/>")
    m = _make_manifest(tmp_path, [theme])
    cfg = _stub_cfg(bebopnet_input_bars="auto", bebopnet_output_bars=8)

    class _FakeCorpus:
        def count_bars(self, p): return 16

    tasks = BebopnetOrchestrator()._build_tasks(cfg, m, out, corpus=_FakeCorpus())
    assert len(tasks) == 1
    assert tasks[0]["input_bars"] == 16
