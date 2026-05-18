"""Загрузчики корпусов для MGEval pipeline.

Walk + chunking вынесены в общий corpus_utils.* — этот модуль остаётся
тонкой обёрткой над ним с конверсией в PrettyMIDI.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import music21 as m21
import pretty_midi

from corpus_utils import (
    extract_melody_part,
    split_into_bar_chunks,
    walk_generated_chunk_files,
)


def _note_count(pm: pretty_midi.PrettyMIDI) -> int:
    return sum(len(inst.notes) for inst in pm.instruments)


def iter_generated_corpus_chunks(
    slug_dir: Path,
    model: str,
    samples_per_theme: int,
    active_themes: list[str],
) -> list[pretty_midi.PrettyMIDI]:
    """Walk по themes/<theme>/<model>/sample_<i>/gen_chunk_<j>.mid.
    Куски с 0 нот скипаются. Несуществующие sample_dir игнорируются.
    """
    chunks: list[pretty_midi.PrettyMIDI] = []
    for path in walk_generated_chunk_files(
        slug_dir, model, samples_per_theme, active_themes, suffix=".mid",
    ):
        pm = pretty_midi.PrettyMIDI(str(path))
        if _note_count(pm) > 0:
            chunks.append(pm)
    return chunks


def load_real_corpus_chunks(
    split_json_path: Path,
    xml_dir: Path,
    chunk_bars: int = 8,
) -> list[pretty_midi.PrettyMIDI]:
    """Загрузить test-соло из split.json, нарезать на непересекающиеся chunk_bars-окна.
    Pickup отбрасывается, хвост короче chunk_bars отбрасывается per-solo,
    куски с 0 нот скипаются.
    """
    test_names = json.loads(split_json_path.read_text())["test"]
    chunks: list[pretty_midi.PrettyMIDI] = []
    for name in test_names:
        xml_path = xml_dir / f"{name}.xml"
        score = m21.converter.parse(str(xml_path))
        melody_part = extract_melody_part(score)
        for chunk_stream in split_into_bar_chunks(melody_part, chunk_bars):
            pm = _score_chunk_to_pretty_midi(chunk_stream)
            if _note_count(pm) > 0:
                chunks.append(pm)
    return chunks


def _score_chunk_to_pretty_midi(chunk: m21.stream.Stream) -> pretty_midi.PrettyMIDI:
    """Экспорт chunk-Stream → midi через temp-файл → PrettyMIDI."""
    mf = m21.midi.translate.streamToMidiFile(chunk)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        mf.open(str(tmp_path), "wb")
        mf.write()
        mf.close()
        return pretty_midi.PrettyMIDI(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
