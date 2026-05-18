"""Тесты shared corpus utilities."""
from __future__ import annotations

from pathlib import Path

import music21 as m21
import pytest

from corpus_utils import (
    extract_melody_part,
    split_into_bar_chunks,
    walk_generated_chunk_files,
)


def _make_score_with_part(notes: list[str]) -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()
    for n in notes:
        part.append(m21.note.Note(n, quarterLength=1.0))
    score.append(part)
    return score


def test_extract_melody_part_single_part_returns_it():
    score = _make_score_with_part(["C4", "E4"])
    part = extract_melody_part(score)
    assert len(part.recurse().notes) == 2


def test_extract_melody_part_picks_first_part_with_notes():
    score = m21.stream.Score()
    empty_part = m21.stream.Part()  # без нот
    score.append(empty_part)
    notes_part = m21.stream.Part()
    notes_part.append(m21.note.Note("G4", quarterLength=1.0))
    score.append(notes_part)
    part = extract_melody_part(score)
    assert len(part.recurse().notes) == 1


def test_extract_melody_part_no_notes_raises():
    score = m21.stream.Score()
    score.append(m21.stream.Part())
    with pytest.raises(ValueError):
        extract_melody_part(score)


def _part_with_bars(
    n_bars: int,
    *,
    pickup: bool = False,
    empty_idx: tuple[int, ...] = (),
) -> m21.stream.Part:
    part = m21.stream.Part()
    if pickup:
        pm0 = m21.stream.Measure(number=0, paddingLeft=3.0)
        pm0.append(m21.note.Note("C4", quarterLength=1.0))
        part.append(pm0)
    for i in range(n_bars):
        m = m21.stream.Measure(number=i + 1)
        if i in empty_idx:
            m.append(m21.note.Rest(quarterLength=4.0))
        else:
            for _ in range(4):
                m.append(m21.note.Note("C4", quarterLength=1.0))
        part.append(m)
    return part


def test_split_into_bar_chunks_full_bars():
    part = _part_with_bars(16)
    chunks = list(split_into_bar_chunks(part, n_bars=8))
    assert len(chunks) == 2
    for c in chunks:
        assert len(list(c.getElementsByClass(m21.stream.Measure))) == 8


def test_split_into_bar_chunks_drops_pickup():
    part = _part_with_bars(16, pickup=True)
    chunks = list(split_into_bar_chunks(part, n_bars=8))
    assert len(chunks) == 2  # pickup отброшен


def test_split_into_bar_chunks_drops_tail():
    part = _part_with_bars(11)
    chunks = list(split_into_bar_chunks(part, n_bars=8))
    assert len(chunks) == 1  # bars 9-11 — tail, dropped


def test_split_into_bar_chunks_shorter_than_one_chunk():
    part = _part_with_bars(5)
    chunks = list(split_into_bar_chunks(part, n_bars=8))
    assert chunks == []


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_walk_collects_active_themes_only(tmp_path):
    _touch(tmp_path / "themes" / "active" / "cmt" / "sample_0" / "gen_chunk_0.mid")
    _touch(tmp_path / "themes" / "inactive" / "cmt" / "sample_0" / "gen_chunk_0.mid")
    paths = list(walk_generated_chunk_files(
        slug_dir=tmp_path, model="cmt",
        samples_per_theme=1, active_themes=["active"], suffix=".mid",
    ))
    assert len(paths) == 1
    assert paths[0].name == "gen_chunk_0.mid"


def test_walk_iterates_all_samples_and_chunks(tmp_path):
    for i in (0, 1):
        for j in (0, 1, 2):
            _touch(tmp_path / "themes" / "T" / "cmt" / f"sample_{i}" / f"gen_chunk_{j}.mid")
    paths = list(walk_generated_chunk_files(
        slug_dir=tmp_path, model="cmt",
        samples_per_theme=2, active_themes=["T"], suffix=".mid",
    ))
    assert len(paths) == 6


def test_walk_missing_sample_dir_ignored(tmp_path):
    paths = list(walk_generated_chunk_files(
        slug_dir=tmp_path, model="cmt",
        samples_per_theme=2, active_themes=["T"], suffix=".mid",
    ))
    assert paths == []


def test_walk_respects_suffix(tmp_path):
    _touch(tmp_path / "themes" / "T" / "cmt" / "sample_0" / "gen_chunk_0.mid")
    _touch(tmp_path / "themes" / "T" / "cmt" / "sample_0" / "gen_chunk_0.musicxml")
    mids = list(walk_generated_chunk_files(
        slug_dir=tmp_path, model="cmt",
        samples_per_theme=1, active_themes=["T"], suffix=".mid",
    ))
    xmls = list(walk_generated_chunk_files(
        slug_dir=tmp_path, model="cmt",
        samples_per_theme=1, active_themes=["T"], suffix=".musicxml",
    ))
    assert len(mids) == 1 and mids[0].suffix == ".mid"
    assert len(xmls) == 1 and xmls[0].suffix == ".musicxml"
