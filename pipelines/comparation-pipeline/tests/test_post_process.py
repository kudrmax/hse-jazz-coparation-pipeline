"""Unit-tests на идемпотентность _post_process_cmt и _post_process_full.

Сценарии:
1. CMT: orphan raw без gen → post_process извлекает gen
2. CMT: всё на месте → post_process не падает, состояние не меняется
3. CMT: битый raw → post_process trash'ит raw, помечает sample fail
4. FULL (mingus): orphan raw_full без gen_full/gen_chunks → всё появляется
5. FULL: всё на месте → no-op
6. FULL: битый raw_full → trash + sample fail
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from comparation_config import (
    BebopnetModelConfig, CmtModelConfig, ComparationConfig,
    MingusModelConfig, SegmentationConfig,
)
from manifest import Manifest
from orchestrators.bebopnet import BebopnetOrchestrator
from orchestrators.cmt import CmtOrchestrator
from orchestrators.mingus import MingusOrchestrator


def _post_process_cmt(manifest, output_dir, results, returncode, tasks, cfg):
    """Bridge-shim: старая сигнатура (tasks ignored)."""
    return CmtOrchestrator()._post_process(manifest, output_dir, results, returncode, cfg)


def _post_process_full(model, manifest, output_dir, results, returncode, tasks, cfg, chunk_bars):
    """Bridge-shim: switch на нужный orchestrator по строке model."""
    orch = {"mingus": MingusOrchestrator(), "bebopnet": BebopnetOrchestrator()}[model]
    return orch._post_process(manifest, output_dir, results, returncode, cfg)

_AUTUMN_8 = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"


def _make_pm(n_bars: int = 16, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """N-bar 4/4 PrettyMIDI, последняя нота за границу +0.01s для надёжности downbeats."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    bar_dur = 60.0 / tempo * 4
    ins = pretty_midi.Instrument(program=0)
    for i in range(n_bars):
        end = (i + 1) * bar_dur
        if i == n_bars - 1:
            end += 0.01
        ins.notes.append(pretty_midi.Note(
            velocity=80, pitch=60 + i % 12, start=i * bar_dur, end=end,
        ))
    pm.instruments.append(ins)
    return pm


def _stub_cfg(samples_per_theme: int = 1) -> ComparationConfig:
    """Минимальный ComparationConfig, достаточный для post_process."""
    return ComparationConfig(
        slug="test", samples_per_theme=samples_per_theme, device="cpu",
        themes_limit="all", output_formats=("midi",),
        segmentation=SegmentationConfig(chunk_bars=8),
        cmt=CmtModelConfig(
            fork_root=Path("/tmp/x"), hparams_yaml_path=Path("/tmp/x"),
            checkpoint_path=Path("/tmp/x"), topk=5, input_bars=8, output_bars=8,
        ),
        mingus=MingusModelConfig(
            fork_root=Path("/tmp/x"), data_path=Path("/tmp/x"),
            checkpoint_dir=Path("/tmp/x"), epochs=10,
            cond_pitch="X", cond_duration="Y", temperature=1.0,
            input_bars="auto", output_bars="auto",
        ),
        bebopnet=BebopnetModelConfig(
            fork_root=Path("/tmp/x"), model_dir=Path("/tmp/x"), checkpoint="x.pt",
            temperature=1.0, input_bars="auto", output_bars="auto",
        ),
    )


def _setup_theme(output_dir: Path, theme: str = "T") -> Path:
    """Создаёт themes/<theme>/ + theme.musicxml + 1 theme_chunk."""
    if not _AUTUMN_8.exists():
        pytest.skip("missing source xml")
    theme_dir = output_dir / "themes" / theme
    theme_dir.mkdir(parents=True)
    (theme_dir / "theme.musicxml").write_bytes(_AUTUMN_8.read_bytes())
    chunks_dir = theme_dir / "theme_chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_0.musicxml").write_bytes(_AUTUMN_8.read_bytes())
    return theme_dir


def _stub_manifest(output_dir: Path, theme: str = "T") -> Manifest:
    m = Manifest(path=output_dir / "manifest.json")
    m.bootstrap("test", "fp", samples_per_theme=1, output_formats=("midi",))
    m.add_theme(theme)
    return m


# ---------- CMT ----------

def test_cmt_extracts_orphan_raw(tmp_path):
    """raw_chunk_0.mid без gen_chunk_0.mid → post_process извлекает gen."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    _make_pm(16).write(str(sample_dir / "raw_chunk_0.mid"))
    assert not (sample_dir / "gen_chunk_0.mid").exists()

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_cmt(m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg)

    assert (sample_dir / "gen_chunk_0.mid").is_file()
    assert m.themes["T"].models["cmt"].samples[0].ok is True


def test_cmt_noop_when_complete(tmp_path):
    """raw + gen на диске → post_process не падает, состояние не меняется."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    _make_pm(16).write(str(sample_dir / "raw_chunk_0.mid"))
    _make_pm(8).write(str(sample_dir / "gen_chunk_0.mid"))
    gen_mtime_before = (sample_dir / "gen_chunk_0.mid").stat().st_mtime

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_cmt(m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg)

    assert (sample_dir / "gen_chunk_0.mid").stat().st_mtime == gen_mtime_before
    assert m.themes["T"].models["cmt"].samples[0].ok is True


def test_cmt_corrupted_raw_trashed_and_failed(tmp_path):
    """Битый raw_chunk_0.mid → trash, sample.ok=False."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "raw_chunk_0.mid").write_bytes(b"NOT_A_MIDI_FILE")

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_cmt(m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg)

    assert not (sample_dir / "raw_chunk_0.mid").exists()
    assert not (sample_dir / "gen_chunk_0.mid").exists()
    s = m.themes["T"].models["cmt"].samples[0]
    assert s.ok is False
    assert "corrupted" in (s.error or "").lower()


# ---------- FULL (mingus/bebopnet) ----------

def test_full_extracts_orphan_raw(tmp_path):
    """raw_full.mid без gen_full/gen_chunks → всё появляется."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "mingus" / "sample_0"
    sample_dir.mkdir(parents=True)
    _make_pm(16).write(str(sample_dir / "raw_full.mid"))
    assert not (sample_dir / "gen_full.mid").exists()
    assert not (sample_dir / "gen_chunk_0.mid").exists()

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_full("mingus", m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg, chunk_bars=8)

    assert (sample_dir / "gen_full.mid").is_file()
    assert (sample_dir / "gen_chunk_0.mid").is_file()
    assert m.themes["T"].models["mingus"].samples[0].ok is True


def test_full_noop_when_complete(tmp_path):
    """raw + gen_full + gen_chunk_0 на диске → no-op."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "mingus" / "sample_0"
    sample_dir.mkdir(parents=True)
    _make_pm(16).write(str(sample_dir / "raw_full.mid"))
    _make_pm(8).write(str(sample_dir / "gen_full.mid"))
    _make_pm(8).write(str(sample_dir / "gen_chunk_0.mid"))
    gen_full_mtime = (sample_dir / "gen_full.mid").stat().st_mtime

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_full("mingus", m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg, chunk_bars=8)

    assert (sample_dir / "gen_full.mid").stat().st_mtime == gen_full_mtime
    assert m.themes["T"].models["mingus"].samples[0].ok is True


def test_full_corrupted_raw_trashed_and_failed(tmp_path):
    """Битый raw_full → trash + sample fail."""
    theme_dir = _setup_theme(tmp_path)
    sample_dir = theme_dir / "bebopnet" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "raw_full.mid").write_bytes(b"NOT_A_MIDI_FILE")

    cfg = _stub_cfg()
    m = _stub_manifest(tmp_path)
    _post_process_full("bebopnet", m, tmp_path, results={}, returncode=0, tasks=[], cfg=cfg, chunk_bars=8)

    assert not (sample_dir / "raw_full.mid").exists()
    assert not (sample_dir / "gen_full.mid").exists()
    s = m.themes["T"].models["bebopnet"].samples[0]
    assert s.ok is False
    assert "corrupted" in (s.error or "").lower()
