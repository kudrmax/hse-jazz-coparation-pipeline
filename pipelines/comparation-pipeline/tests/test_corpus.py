"""Tests for Corpus.active_subset() and count_bars()."""
from __future__ import annotations

from pathlib import Path

import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from corpus import Corpus


def test_active_subset_all(tmp_path):
    """Если themes_limit=='all' — возвращает весь sorted список."""
    (tmp_path / "B.musicxml").write_text("dummy")
    (tmp_path / "A.musicxml").write_text("dummy")
    (tmp_path / "C.musicxml").write_text("dummy")
    corpus = Corpus(root=tmp_path)
    paths = corpus.active_subset("all")
    assert [p.stem for p in paths] == ["A", "B", "C"]


def test_active_subset_limit(tmp_path):
    for n in ["A", "B", "C", "D", "E"]:
        (tmp_path / f"{n}.musicxml").write_text("dummy")
    corpus = Corpus(root=tmp_path)
    paths = corpus.active_subset(2)
    assert [p.stem for p in paths] == ["A", "B"]


def test_active_subset_limit_more_than_corpus_returns_all(tmp_path):
    (tmp_path / "A.musicxml").write_text("dummy")
    corpus = Corpus(root=tmp_path)
    paths = corpus.active_subset(10)
    assert len(paths) == 1


def test_active_subset_recursive(tmp_path):
    """Корпус может иметь вложенные подкаталоги."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "Z.musicxml").write_text("dummy")
    (tmp_path / "A.musicxml").write_text("dummy")
    corpus = Corpus(root=tmp_path)
    paths = corpus.active_subset("all")
    assert len(paths) == 2  # rglob


def test_count_bars(tmp_path):
    """Использует m21 на реальном файле."""
    real_xml = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"
    if not real_xml.exists():
        pytest.skip("Autumn_Leaves_8bars.musicxml missing")
    corpus = Corpus(root=real_xml.parent)
    n = corpus.count_bars(real_xml)
    assert n == 8
