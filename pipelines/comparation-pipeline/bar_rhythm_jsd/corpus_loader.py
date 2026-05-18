"""Загрузчики корпусов для bar-rhythm-jsd (Measure-уровень)."""
from __future__ import annotations

import json
from pathlib import Path

import music21 as m21

from corpus_utils import (
    extract_melody_part,
    split_into_bar_chunks,
    walk_generated_chunk_files,
)


def load_real_corpus_measures(
    split_json_path: Path,
    xml_dir: Path,
    chunk_bars: int = 8,
) -> list[m21.stream.Measure]:
    """Загрузить WjazzD test-соло, нарезать на chunk_bars-окна, вернуть
    плоский список Measure'ов из всех окон.

    Pickup и хвост короче chunk_bars дропаются (через split_into_bar_chunks).
    """
    test_names = json.loads(split_json_path.read_text())["test"]
    measures: list[m21.stream.Measure] = []
    for name in test_names:
        xml_path = xml_dir / f"{name}.xml"
        score = m21.converter.parse(str(xml_path))
        melody_part = extract_melody_part(score)
        for chunk in split_into_bar_chunks(melody_part, chunk_bars):
            measures.extend(chunk.getElementsByClass(m21.stream.Measure))
    return measures


def iter_generated_corpus_measures(
    slug_dir: Path,
    model: str,
    samples_per_theme: int,
    active_themes: list[str],
) -> list[m21.stream.Measure]:
    """Walk по themes/<theme>/<model>/sample_<i>/gen_chunk_<j>.musicxml,
    распарсить каждый xml, вернуть flatten-список всех Measure'ов.

    Score без Measure'ов — log + skip.
    """
    measures: list[m21.stream.Measure] = []
    for path in walk_generated_chunk_files(
        slug_dir, model, samples_per_theme, active_themes, suffix=".musicxml",
    ):
        score = m21.converter.parse(str(path))
        try:
            melody_part = extract_melody_part(score)
        except ValueError:
            print(f"warn: no melody part in {path}, skipping", flush=True)
            continue
        bar_list = list(melody_part.getElementsByClass(m21.stream.Measure))
        if not bar_list:
            print(f"warn: no measures in {path}, skipping", flush=True)
            continue
        measures.extend(bar_list)
    return measures
