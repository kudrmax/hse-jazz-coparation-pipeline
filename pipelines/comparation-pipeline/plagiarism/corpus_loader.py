"""Загрузчики корпусов для plagiarism: train (WjazzD train=344 без chunking)
и generated (gen_chunk_*.mid файлы из themes/<theme>/<model>/sample_<i>/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import music21 as m21
import pretty_midi

from corpus_utils import walk_generated_chunk_files

from .intervals import intervals_from_midi, intervals_from_score


def load_train_corpus_intervals(
    split_json_path: Path,
    xml_dir: Path,
) -> list[list[int]]:
    """Читает split.json["train"], парсит каждое <name>.xml, возвращает
    интервал-последовательность per соло.

    Train не нарезается на chunks — каждое соло как одна list[int].
    Соло с <2 нот / parse error / отсутствующим xml → warning в stderr + skip.

    Возвращает list[list[int]], длина ≤ |split["train"]|.
    """
    train_names = json.loads(split_json_path.read_text())["train"]
    train_intervals: list[list[int]] = []
    for name in train_names:
        xml_path = xml_dir / f"{name}.xml"
        if not xml_path.exists():
            print(f"plagiarism: train xml missing, skip: {name}", file=sys.stderr)
            continue
        try:
            score = m21.converter.parse(str(xml_path))
        except Exception as e:
            print(f"plagiarism: train parse failed for {name}: {e}", file=sys.stderr)
            continue
        intervals = intervals_from_score(score)
        if not intervals:
            print(f"plagiarism: train solo too short (<2 notes), skip: {name}", file=sys.stderr)
            continue
        train_intervals.append(intervals)
    return train_intervals


def iter_generated_corpus_intervals(
    slug_dir: Path,
    model: str,
    samples_per_theme: int,
    active_themes: list[str],
) -> list[list[int]]:
    """Walk через themes/<theme>/<model>/sample_<i>/gen_chunk_<j>.mid,
    конвертируется каждый MIDI в list[int] интервалов.

    Возвращает list[list[int]] — по элементу на chunk (включая пустые;
    фильтрация уровня pipeline).
    """
    out: list[list[int]] = []
    for path in walk_generated_chunk_files(
        slug_dir, model, samples_per_theme, active_themes, suffix=".mid",
    ):
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception as e:
            print(f"plagiarism: midi parse failed for {path}: {e}", file=sys.stderr)
            out.append([])
            continue
        out.append(intervals_from_midi(pm))
    return out
