"""Tests for CmtOrchestrator."""
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
from orchestrators.cmt import CmtOrchestrator


def _stub_cfg() -> ComparationConfig:
    return ComparationConfig(
        slug="t", samples_per_theme=1, device="cpu",
        themes_limit="all", output_formats=("midi",),
        segmentation=SegmentationConfig(chunk_bars=8),
        cmt=CmtModelConfig(
            fork_root=Path("/cmt/fork"),
            hparams_yaml_path=Path("/cmt/hp.yaml"),
            checkpoint_path=Path("/cmt/ckpt"),
            topk=5, input_bars=8, output_bars=8,
        ),
        mingus=MingusModelConfig(
            fork_root=Path("/m"), data_path=Path("/m/data"),
            checkpoint_dir=Path("/m/ck"), epochs=10,
            cond_pitch="D-C-B-BE-O", cond_duration="B-BE-O",
            temperature=1.0, input_bars=8, output_bars=8,
        ),
        bebopnet=BebopnetModelConfig(
            fork_root=Path("/b"), model_dir=Path("/b/md"),
            checkpoint="ck.pt", temperature=1.0,
            input_bars=8, output_bars=8,
        ),
    )


def test_cmt_name():
    assert CmtOrchestrator().name == "cmt"


def test_cmt_derived_yaml_block():
    cfg = _stub_cfg()
    block = CmtOrchestrator()._derived_yaml_block(cfg)
    assert block["model"] == "cmt"
    assert block["output"] == {"formats": ["midi"], "force_overwrite": True}
    assert block["common"] == {
        "seed": 1, "input_bars": 8, "output_bars": 8, "device": "cpu",
    }
    assert block["cmt"]["fork_root"] == "/cmt/fork"
    assert block["cmt"]["topk"] == 5


def test_cmt_sample_complete_all_chunks_present(tmp_path):
    sd = tmp_path / "s0"
    sd.mkdir()
    for j in range(3):
        (sd / f"raw_chunk_{j}.mid").write_bytes(b"x")
    assert CmtOrchestrator().sample_complete(sd, n_chunks=3) is True


def test_cmt_sample_complete_missing_chunk(tmp_path):
    sd = tmp_path / "s0"
    sd.mkdir()
    (sd / "raw_chunk_0.mid").write_bytes(b"x")
    (sd / "raw_chunk_2.mid").write_bytes(b"x")  # пропуск j=1
    assert CmtOrchestrator().sample_complete(sd, n_chunks=3) is False


def test_cmt_sample_complete_missing_dir(tmp_path):
    assert CmtOrchestrator().sample_complete(tmp_path / "nope", n_chunks=3) is False


from manifest import Manifest  # noqa: E402


def _make_manifest(tmp_path, themes: list[str], samples_per_theme: int = 1) -> Manifest:
    m = Manifest(path=tmp_path / "manifest.json", samples_per_theme=samples_per_theme)
    for t in themes:
        m.add_theme(t)
    return m


def test_cmt_build_tasks_skips_if_raw_chunk_exists(tmp_path):
    """Если raw_chunk_<j>.mid уже есть — task под этот chunk не строится."""
    theme = "T"
    output_dir = tmp_path / "out"
    themes_root = output_dir / "themes"
    theme_dir = themes_root / theme
    chunks_dir = theme_dir / "theme_chunks"
    chunks_dir.mkdir(parents=True)
    for j in range(2):
        (chunks_dir / f"chunk_{j}.musicxml").write_text("<x/>")
    # raw_chunk_0 уже есть
    sample_dir = theme_dir / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "raw_chunk_0.mid").write_bytes(b"x")

    m = _make_manifest(tmp_path, [theme])
    cfg = _stub_cfg()

    tasks = CmtOrchestrator()._build_tasks(cfg, m, output_dir, corpus=None)
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "T/0/chunk_1"
    assert tasks[0]["seed"] == 42  # SEED_BASE + idx=0


def test_cmt_build_tasks_skips_removed_from_corpus(tmp_path):
    theme = "T"
    output_dir = tmp_path / "out"
    (output_dir / "themes" / theme / "theme_chunks").mkdir(parents=True)
    (output_dir / "themes" / theme / "theme_chunks" / "chunk_0.musicxml").write_text("<x/>")
    m = _make_manifest(tmp_path, [theme])
    m.themes[theme].removed_from_corpus = True
    cfg = _stub_cfg()
    assert CmtOrchestrator()._build_tasks(cfg, m, output_dir, corpus=None) == []


def test_cmt_build_tasks_skips_sample_already_ok(tmp_path):
    """Если manifest sample.ok=True — task не строится даже если raw отсутствует."""
    theme = "T"
    output_dir = tmp_path / "out"
    chunks_dir = output_dir / "themes" / theme / "theme_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_0.musicxml").write_text("<x/>")
    m = _make_manifest(tmp_path, [theme])
    m.mark_sample(theme, "cmt", 0, ok=True)
    cfg = _stub_cfg()
    assert CmtOrchestrator()._build_tasks(cfg, m, output_dir, corpus=None) == []
