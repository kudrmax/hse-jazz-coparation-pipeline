"""Тесты corpus loaders для plagiarism (train без chunking, gen-chunks)."""
from __future__ import annotations

import json
from pathlib import Path

import music21 as m21
import pretty_midi

from plagiarism.corpus_loader import (
    iter_generated_corpus_intervals,
    load_train_corpus_intervals,
)


def _write_train_xml(xml_dir: Path, name: str, pitches: list[str]) -> None:
    score = m21.stream.Score()
    part = m21.stream.Part()
    m = m21.stream.Measure(number=1)
    m.append(m21.meter.TimeSignature("4/4"))
    for p in pitches:
        m.append(m21.note.Note(p, quarterLength=1.0))
    part.append(m)
    score.append(part)
    xml_dir.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(xml_dir / f"{name}.xml"))


def _write_gen_mid(path: Path, pitches: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=i * 0.5, end=(i + 1) * 0.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_load_train_uses_train_key_not_test(tmp_path):
    """Loader читает split.json['train'], НЕ ['test']."""
    xml_dir = tmp_path / "xml"
    _write_train_xml(xml_dir, "solo_train", ["C4", "E4", "G4"])
    _write_train_xml(xml_dir, "solo_test", ["C4", "D4", "E4"])

    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "train": ["solo_train"],
        "test": ["solo_test"],
    }))

    train = load_train_corpus_intervals(split, xml_dir)
    # должны быть только train-интервалы — [4, 3]
    assert train == [[4, 3]]


def test_load_train_does_not_chunk(tmp_path):
    """Train-соло НЕ нарезается на bar-окна — одно соло → одна list[int]."""
    xml_dir = tmp_path / "xml"
    # 16-bar соло → одна интервал-последовательность длины 16*4-1=63
    pitches = ["C4", "D4", "E4", "F4"] * 16
    _write_train_xml(xml_dir, "solo16", pitches)

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["solo16"]}))

    train = load_train_corpus_intervals(split, xml_dir)
    assert len(train) == 1  # одно соло, не несколько chunks
    assert len(train[0]) == 63  # 64 ноты → 63 интервалов


def test_load_train_skips_missing_file(tmp_path, capsys):
    """Отсутствующий xml-файл → warning + skip, не падение."""
    xml_dir = tmp_path / "xml"
    _write_train_xml(xml_dir, "solo_ok", ["C4", "E4", "G4"])

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["solo_ok", "solo_missing"]}))

    train = load_train_corpus_intervals(split, xml_dir)
    assert train == [[4, 3]]  # только solo_ok
    captured = capsys.readouterr()
    assert "solo_missing" in (captured.out + captured.err)


def test_load_train_skips_short_solo(tmp_path):
    """Соло с <2 нот не даёт интервалов → skip."""
    xml_dir = tmp_path / "xml"
    _write_train_xml(xml_dir, "solo_short", ["C4"])  # 1 нота
    _write_train_xml(xml_dir, "solo_ok", ["C4", "E4"])

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["solo_short", "solo_ok"]}))

    train = load_train_corpus_intervals(split, xml_dir)
    assert train == [[4]]


def test_iter_generated_corpus_intervals_walks_mid(tmp_path):
    slug_dir = tmp_path / "slug"
    _write_gen_mid(
        slug_dir / "themes" / "T1" / "cmt" / "sample_0" / "gen_chunk_0.mid",
        [60, 62, 64, 67],
    )
    _write_gen_mid(
        slug_dir / "themes" / "T1" / "cmt" / "sample_0" / "gen_chunk_1.mid",
        [60, 65],
    )
    intervals = iter_generated_corpus_intervals(
        slug_dir, "cmt", samples_per_theme=1, active_themes=["T1"],
    )
    assert intervals == [[2, 2, 3], [5]]


def test_iter_generated_corpus_intervals_inactive_themes(tmp_path):
    slug_dir = tmp_path / "slug"
    _write_gen_mid(
        slug_dir / "themes" / "Inactive" / "cmt" / "sample_0" / "gen_chunk_0.mid",
        [60, 62, 64],
    )
    intervals = iter_generated_corpus_intervals(
        slug_dir, "cmt", samples_per_theme=1, active_themes=["Other"],
    )
    assert intervals == []


def test_iter_generated_corpus_intervals_multiple_samples(tmp_path):
    slug_dir = tmp_path / "slug"
    for s in range(2):
        _write_gen_mid(
            slug_dir / "themes" / "T1" / "cmt" / f"sample_{s}" / "gen_chunk_0.mid",
            [60, 62 + s, 64],
        )
    intervals = iter_generated_corpus_intervals(
        slug_dir, "cmt", samples_per_theme=2, active_themes=["T1"],
    )
    # sample_0: [60,62,64] → [2,2]; sample_1: [60,63,64] → [3,1]
    assert intervals == [[2, 2], [3, 1]]
