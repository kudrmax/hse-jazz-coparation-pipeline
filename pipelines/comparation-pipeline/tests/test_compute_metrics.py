"""Tests for compute_metrics.py — walk gen_chunk_<j>.mid + theme_chunks/chunk_<j>.musicxml."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

_AUTUMN_8 = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"


def _make_chunk_mid(path: Path) -> None:
    """8-bar 4/4 PrettyMIDI с 16 нотами (по 0.5 секунды каждая, 120 BPM)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    ins = pretty_midi.Instrument(program=0)
    for i in range(16):
        ins.notes.append(pretty_midi.Note(
            velocity=80, pitch=60 + i % 12, start=i * 1.0, end=(i + 1) * 1.0,
        ))
    pm.instruments.append(ins)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def _stub_layout(tmp_path: Path) -> Path:
    """Создаёт outputs/x/themes/{AllOk,Bad}/ с новой структурой:
    theme_chunks/chunk_0.musicxml + <model>/sample_<i>/gen_chunk_0.mid."""
    if not _AUTUMN_8.exists():
        pytest.skip("missing source xml")
    out = tmp_path / "outputs" / "x"
    themes = out / "themes"

    for theme in ("AllOk", "Bad"):
        theme_dir = themes / theme
        theme_dir.mkdir(parents=True)
        (theme_dir / "theme.musicxml").write_bytes(_AUTUMN_8.read_bytes())
        chunks_dir = theme_dir / "theme_chunks"
        chunks_dir.mkdir()
        (chunks_dir / "chunk_0.musicxml").write_bytes(_AUTUMN_8.read_bytes())

        for model in ("cmt", "mingus", "bebopnet"):
            for idx in range(2):
                if theme == "Bad" and model == "cmt":
                    continue  # cmt на Bad не дал чанков
                _make_chunk_mid(theme_dir / model / f"sample_{idx}" / "gen_chunk_0.mid")

    manifest = {
        "config_slug": "x", "config_fingerprint": "fp",
        "samples_per_theme": 2, "output_formats": ["midi"],
        "started_at": "2026-05-09T12:00:00Z",
        "last_updated_at": "2026-05-09T12:00:00Z",
        "themes": {
            "AllOk": {
                "status": "ok", "removed_from_corpus": False, "excluded_due_to": None,
                "models": {
                    m: {"status": "ok", "samples": {"0": {"ok": True}, "1": {"ok": True}}}
                    for m in ("cmt", "mingus", "bebopnet")
                },
            },
            "Bad": {
                "status": "fail", "removed_from_corpus": False, "excluded_due_to": "cmt",
                "models": {
                    "cmt": {"status": "fail", "samples": {"0": {"ok": False, "error": "boom"}}},
                    "mingus": {"status": "ok", "samples": {"0": {"ok": True}, "1": {"ok": True}}},
                    "bebopnet": {"status": "ok", "samples": {"0": {"ok": True}, "1": {"ok": True}}},
                },
            },
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_only_ok_themes_in_per_segment(tmp_path):
    out = _stub_layout(tmp_path)
    from manifest import Manifest
    from phases.metrics import PerSegmentMetricsRunner
    sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline/metrics"))
    from registry import all_metrics  # type: ignore[import-not-found]
    m = Manifest.load(out / "manifest.json")
    PerSegmentMetricsRunner(metrics=all_metrics()).run(out, m, chunk_bars=8)

    rows = list(csv.DictReader((out / "_metrics" / "per_segment.csv").open()))
    themes = {r["theme"] for r in rows}
    assert themes == {"AllOk"}


def test_per_segment_row_count(tmp_path):
    """1 ok-theme × 3 models × 2 samples × 1 chunk = 6 rows."""
    out = _stub_layout(tmp_path)
    from manifest import Manifest
    from phases.metrics import PerSegmentMetricsRunner
    sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline/metrics"))
    from registry import all_metrics  # type: ignore[import-not-found]
    m = Manifest.load(out / "manifest.json")
    PerSegmentMetricsRunner(metrics=all_metrics()).run(out, m, chunk_bars=8)
    rows = list(csv.DictReader((out / "_metrics" / "per_segment.csv").open()))
    assert len(rows) == 6


def test_three_csvs_created(tmp_path):
    out = _stub_layout(tmp_path)
    from manifest import Manifest
    from phases.metrics import PerSegmentMetricsRunner
    sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline/metrics"))
    from registry import all_metrics  # type: ignore[import-not-found]
    m = Manifest.load(out / "manifest.json")
    PerSegmentMetricsRunner(metrics=all_metrics()).run(out, m, chunk_bars=8)
    metrics_dir = out / "_metrics"
    assert (metrics_dir / "per_segment.csv").is_file()
    assert (metrics_dir / "aggregates.csv").is_file()
    assert (metrics_dir / "significance.csv").is_file()
