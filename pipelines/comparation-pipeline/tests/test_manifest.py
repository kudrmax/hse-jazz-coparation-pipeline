"""Tests for Manifest class — single JSON state-file with atomic save."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from manifest import Manifest


def test_create_empty(tmp_path):
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap(
        config_slug="test",
        config_fingerprint="sha256:abc",
        samples_per_theme=2,
        output_formats=("midi",),
    )
    assert m.themes == {}
    m.save_atomic()
    assert (tmp_path / "manifest.json").is_file()


def test_load_existing(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({
        "config_slug": "x",
        "config_fingerprint": "sha256:abc",
        "samples_per_theme": 2,
        "output_formats": ["midi"],
        "started_at": "2026-05-09T12:00:00Z",
        "last_updated_at": "2026-05-09T12:00:00Z",
        "themes": {},
    }))
    m = Manifest.load(p)
    assert m.config_slug == "x"
    assert m.samples_per_theme == 2


def test_mark_sample_ok_and_recompute(tmp_path):
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("x", "fp", samples_per_theme=2, output_formats=("midi",))
    m.add_theme("AllOfMe_C")
    m.mark_sample("AllOfMe_C", "cmt", 0, ok=True, duration_sec=8.2)
    m.mark_sample("AllOfMe_C", "cmt", 1, ok=True, duration_sec=7.9)
    m.mark_sample("AllOfMe_C", "mingus", 0, ok=True, duration_sec=10.1)
    m.mark_sample("AllOfMe_C", "mingus", 1, ok=True, duration_sec=10.0)
    m.mark_sample("AllOfMe_C", "bebopnet", 0, ok=True, duration_sec=11.5)
    m.mark_sample("AllOfMe_C", "bebopnet", 1, ok=True, duration_sec=11.4)
    m.recompute_derived_status()
    t = m.themes["AllOfMe_C"]
    assert t.status == "ok"
    assert t.models["cmt"].status == "ok"


def test_one_failed_sample_marks_model_and_theme_fail(tmp_path):
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("x", "fp", samples_per_theme=2, output_formats=("midi",))
    m.add_theme("Caravan")
    m.mark_sample("Caravan", "cmt", 0, ok=True, duration_sec=8.0)
    m.mark_sample("Caravan", "cmt", 1, ok=False, error="ChordSymbol 'C7alt'")
    m.mark_sample("Caravan", "mingus", 0, ok=True, duration_sec=10.0)
    m.mark_sample("Caravan", "mingus", 1, ok=True, duration_sec=10.0)
    m.mark_sample("Caravan", "bebopnet", 0, ok=True, duration_sec=11.0)
    m.mark_sample("Caravan", "bebopnet", 1, ok=True, duration_sec=11.0)
    m.recompute_derived_status()
    t = m.themes["Caravan"]
    assert t.status == "fail"
    assert t.models["cmt"].status == "fail"
    assert t.excluded_due_to == "cmt"
    assert t.models["mingus"].status == "ok"


def test_atomic_save_uses_tmp_rename(tmp_path):
    """Sanity: save через tmp файл (даже если crashed mid-save, manifest.json валидный)."""
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("x", "fp", samples_per_theme=1, output_formats=("midi",))
    m.add_theme("AllOfMe_C")
    m.mark_sample("AllOfMe_C", "cmt", 0, ok=True, duration_sec=1.0)
    m.save_atomic()
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert raw["themes"]["AllOfMe_C"]["models"]["cmt"]["samples"]["0"]["ok"]


def test_failed_pairs_returns_only_failed(tmp_path):
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("x", "fp", samples_per_theme=1, output_formats=("midi",))
    m.add_theme("AllOfMe_C")
    m.add_theme("Caravan")
    m.mark_sample("AllOfMe_C", "cmt", 0, ok=True, duration_sec=1.0)
    m.mark_sample("AllOfMe_C", "mingus", 0, ok=True, duration_sec=1.0)
    m.mark_sample("AllOfMe_C", "bebopnet", 0, ok=True, duration_sec=1.0)
    m.mark_sample("Caravan", "cmt", 0, ok=False, error="boom")
    m.recompute_derived_status()
    pairs = m.failed_pairs()
    assert ("Caravan", "cmt", 0) in pairs
    assert all(p[0] != "AllOfMe_C" for p in pairs)


def test_removed_from_corpus_blocks_theme_status_ok(tmp_path):
    """Если тема помечена removed_from_corpus=True — её status всегда fail
    даже если все sample'ы ok."""
    m = Manifest(path=tmp_path / "manifest.json")
    m.bootstrap("x", "fp", samples_per_theme=1, output_formats=("midi",))
    m.add_theme("Old")
    m.mark_sample("Old", "cmt", 0, ok=True, duration_sec=1.0)
    m.mark_sample("Old", "mingus", 0, ok=True, duration_sec=1.0)
    m.mark_sample("Old", "bebopnet", 0, ok=True, duration_sec=1.0)
    m.themes["Old"].removed_from_corpus = True
    m.recompute_derived_status()
    assert m.themes["Old"].status == "fail"


def test_round_trip_through_save_and_load(tmp_path):
    p = tmp_path / "manifest.json"
    m = Manifest(path=p)
    m.bootstrap("rt", "sha256:rt", samples_per_theme=2, output_formats=("midi", "musicxml"))
    m.add_theme("X")
    m.mark_sample("X", "cmt", 0, ok=True, duration_sec=2.5)
    m.mark_sample("X", "cmt", 1, ok=False, error="boom")
    m.recompute_derived_status()
    m.save_atomic()

    m2 = Manifest.load(p)
    assert m2.config_slug == "rt"
    assert m2.samples_per_theme == 2
    assert m2.output_formats == ("midi", "musicxml")
    assert m2.themes["X"].models["cmt"].samples[0].ok is True
    assert m2.themes["X"].models["cmt"].samples[1].error == "boom"


def test_load_nonexistent_returns_empty_manifest(tmp_path):
    m = Manifest.load(tmp_path / "absent.json")
    assert m.config_slug == ""
    assert m.themes == {}


def test_active_themes_filters_status_and_removed():
    """active_themes() возвращает только темы со status=ok и not removed_from_corpus."""
    from manifest import ThemeStatus

    m = Manifest(path=Path("/dev/null"))
    m.themes = {
        "ok_kept": ThemeStatus(status="ok", removed_from_corpus=False),
        "ok_removed": ThemeStatus(status="ok", removed_from_corpus=True),
        "fail_kept": ThemeStatus(status="fail", removed_from_corpus=False),
        "fail_removed": ThemeStatus(status="fail", removed_from_corpus=True),
    }
    assert m.active_themes() == ["ok_kept"]
