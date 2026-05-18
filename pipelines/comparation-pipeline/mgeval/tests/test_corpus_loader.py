"""Тесты corpus_loader."""
import json
from pathlib import Path

import music21 as m21
import pretty_midi

from mgeval.corpus_loader import (
    iter_generated_corpus_chunks, load_real_corpus_chunks,
)
from mgeval.tests._helpers import mk_pm


def _make_solo_xml(
    tmp_dir: Path,
    name: str,
    n_full_bars: int,
    *,
    add_pickup: bool = False,
    empty_bar_indices: tuple[int, ...] = (),
) -> Path:
    """Записать synthetic xml: optional 1-beat pickup + n_full_bars 4/4 четвертей.
    empty_bar_indices: 0-based индексы баров без нот (для теста zero-note chunk).
    """
    part = m21.stream.Part()
    if add_pickup:
        pickup = m21.stream.Measure(number=0, paddingLeft=3.0)
        pickup.append(m21.meter.TimeSignature("4/4"))
        pickup.append(m21.note.Note("C4", quarterLength=1.0))
        part.append(pickup)
    for bar_idx in range(n_full_bars):
        measure = m21.stream.Measure(number=bar_idx + 1)
        if bar_idx == 0 and not add_pickup:
            measure.append(m21.meter.TimeSignature("4/4"))
        if bar_idx in empty_bar_indices:
            measure.append(m21.note.Rest(quarterLength=4.0))
        else:
            for _ in range(4):
                measure.append(m21.note.Note("C4", quarterLength=1.0))
        part.append(measure)
    score = m21.stream.Score()
    score.append(part)
    out = tmp_dir / f"{name}.xml"
    score.write("musicxml", fp=str(out))
    return out


def _write_split_json(tmp_dir: Path, test_names: list[str]) -> Path:
    path = tmp_dir / "split.json"
    path.write_text(json.dumps({"test": test_names}))
    return path


def _write_midi(path: Path, pm: pretty_midi.PrettyMIDI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def test_iter_generated_corpus_chunks_smoke(tmp_path):
    pm_with_note = mk_pm([(60, 0.0, 1.0)])
    _write_midi(
        tmp_path / "themes" / "theme1" / "cmt" / "sample_0" / "gen_chunk_0.mid",
        pm_with_note,
    )
    chunks = iter_generated_corpus_chunks(
        slug_dir=tmp_path,
        model="cmt",
        samples_per_theme=1,
        active_themes=["theme1"],
    )
    assert len(chunks) == 1


def test_iter_generated_corpus_chunks_zero_note_skipped(tmp_path):
    pm_empty = mk_pm([])
    _write_midi(
        tmp_path / "themes" / "theme1" / "cmt" / "sample_0" / "gen_chunk_0.mid",
        pm_empty,
    )
    chunks = iter_generated_corpus_chunks(
        slug_dir=tmp_path,
        model="cmt",
        samples_per_theme=1,
        active_themes=["theme1"],
    )
    assert chunks == []


def test_iter_generated_corpus_chunks_inactive_theme_skipped(tmp_path):
    pm = mk_pm([(60, 0.0, 1.0)])
    _write_midi(
        tmp_path / "themes" / "theme_inactive" / "cmt" / "sample_0" / "gen_chunk_0.mid",
        pm,
    )
    chunks = iter_generated_corpus_chunks(
        slug_dir=tmp_path,
        model="cmt",
        samples_per_theme=1,
        active_themes=["theme_other"],
    )
    assert chunks == []


def test_iter_generated_corpus_chunks_missing_sample_dir_ignored(tmp_path):
    chunks = iter_generated_corpus_chunks(
        slug_dir=tmp_path,
        model="cmt",
        samples_per_theme=2,
        active_themes=["nonexistent"],
    )
    assert chunks == []


def test_real_corpus_loader_smoke(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _make_solo_xml(xml_dir, "solo1", n_full_bars=16)
    split_path = _write_split_json(tmp_path, ["solo1"])
    chunks = load_real_corpus_chunks(
        split_json_path=split_path,
        xml_dir=xml_dir,
        chunk_bars=8,
    )
    assert len(chunks) == 2  # 16 bars / 8 = 2 chunks


def test_real_corpus_loader_anacrusis_dropped(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _make_solo_xml(xml_dir, "solo_pickup", n_full_bars=16, add_pickup=True)
    split_path = _write_split_json(tmp_path, ["solo_pickup"])
    chunks = load_real_corpus_chunks(
        split_json_path=split_path,
        xml_dir=xml_dir,
        chunk_bars=8,
    )
    # 16 full bars (pickup dropped) → 2 chunks
    assert len(chunks) == 2


def test_real_corpus_loader_tail_dropped(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _make_solo_xml(xml_dir, "solo_11", n_full_bars=11)
    split_path = _write_split_json(tmp_path, ["solo_11"])
    chunks = load_real_corpus_chunks(
        split_json_path=split_path,
        xml_dir=xml_dir,
        chunk_bars=8,
    )
    # 11 bars → 1 chunk (bars 1-8), bars 9-11 dropped as tail
    assert len(chunks) == 1


def test_real_corpus_loader_zero_note_chunk_skipped(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    # 16 bars total, второй 8-блок (0-based bars 8-15) пустой
    _make_solo_xml(
        xml_dir, "solo_empty",
        n_full_bars=16,
        empty_bar_indices=tuple(range(8, 16)),
    )
    split_path = _write_split_json(tmp_path, ["solo_empty"])
    chunks = load_real_corpus_chunks(
        split_json_path=split_path,
        xml_dir=xml_dir,
        chunk_bars=8,
    )
    # Только первый chunk имеет ноты → 1 chunk
    assert len(chunks) == 1


def test_real_corpus_loader_solo_shorter_than_chunk_returns_nothing(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _make_solo_xml(xml_dir, "solo_5", n_full_bars=5)
    split_path = _write_split_json(tmp_path, ["solo_5"])
    chunks = load_real_corpus_chunks(
        split_json_path=split_path,
        xml_dir=xml_dir,
        chunk_bars=8,
    )
    assert chunks == []
