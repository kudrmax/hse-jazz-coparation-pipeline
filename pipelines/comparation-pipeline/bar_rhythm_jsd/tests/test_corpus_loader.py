"""Тесты corpus_loader (Measure-уровневый)."""
from __future__ import annotations

import json
from pathlib import Path

import music21 as m21

from bar_rhythm_jsd.corpus_loader import (
    iter_generated_corpus_measures,
    load_real_corpus_measures,
)


def _make_solo_xml(xml_dir: Path, name: str, n_full_bars: int) -> None:
    score = m21.stream.Score()
    part = m21.stream.Part()
    for i in range(n_full_bars):
        m = m21.stream.Measure(number=i + 1)
        if i == 0:
            m.append(m21.meter.TimeSignature("4/4"))
        for _ in range(4):
            m.append(m21.note.Note("C4", quarterLength=1.0))
        part.append(m)
    score.append(part)
    xml_dir.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(xml_dir / f"{name}.xml"))


def test_load_real_corpus_measures_flattens_chunks(tmp_path):
    xml_dir = tmp_path / "xml"
    _make_solo_xml(xml_dir, "solo16", n_full_bars=16)  # 2 chunks × 8 bars
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["solo16"]}))

    measures = load_real_corpus_measures(split, xml_dir, chunk_bars=8)
    assert len(measures) == 16  # 2 chunks × 8 bars
    for m in measures:
        assert isinstance(m, m21.stream.Measure)


def test_load_real_corpus_measures_drops_tail(tmp_path):
    xml_dir = tmp_path / "xml"
    _make_solo_xml(xml_dir, "solo11", n_full_bars=11)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["solo11"]}))

    measures = load_real_corpus_measures(split, xml_dir, chunk_bars=8)
    assert len(measures) == 8  # один chunk, tail отрезан


def _write_gen_chunk_xml(path: Path, n_bars: int) -> None:
    score = m21.stream.Score()
    part = m21.stream.Part()
    for i in range(n_bars):
        m = m21.stream.Measure(number=i + 1)
        if i == 0:
            m.append(m21.meter.TimeSignature("4/4"))
        m.append(m21.note.Note("C4", quarterLength=4.0))
        part.append(m)
    score.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))


def test_iter_generated_corpus_measures_smoke(tmp_path):
    slug_dir = tmp_path / "slug"
    _write_gen_chunk_xml(
        slug_dir / "themes" / "T1" / "cmt" / "sample_0" / "gen_chunk_0.musicxml",
        n_bars=8,
    )
    measures = iter_generated_corpus_measures(
        slug_dir, "cmt", samples_per_theme=1, active_themes=["T1"],
    )
    assert len(measures) == 8


def test_iter_generated_corpus_measures_inactive_skipped(tmp_path):
    slug_dir = tmp_path / "slug"
    _write_gen_chunk_xml(
        slug_dir / "themes" / "Inactive" / "cmt" / "sample_0" / "gen_chunk_0.musicxml",
        n_bars=8,
    )
    measures = iter_generated_corpus_measures(
        slug_dir, "cmt", samples_per_theme=1, active_themes=["Other"],
    )
    assert measures == []


def test_iter_generated_corpus_measures_multiple_chunks(tmp_path):
    slug_dir = tmp_path / "slug"
    for j in range(3):
        _write_gen_chunk_xml(
            slug_dir / "themes" / "T1" / "cmt" / "sample_0" / f"gen_chunk_{j}.musicxml",
            n_bars=8,
        )
    measures = iter_generated_corpus_measures(
        slug_dir, "cmt", samples_per_theme=1, active_themes=["T1"],
    )
    assert len(measures) == 24
