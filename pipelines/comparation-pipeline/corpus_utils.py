"""Общие утилиты для distributional metric pipelines (MGEval, Bar-Rhythm-JSD).

Walk по generated outputs и нарезка real WjazzD на N-bar окна.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import music21 as m21


def extract_melody_part(score: m21.stream.Score) -> m21.stream.Part:
    """Первый Part с notes > 0 (защита от score'ов с отдельным chord-part'ом)."""
    for part in score.parts:
        if len(part.recurse().notes) > 0:
            return part
    raise ValueError("Score has no part with notes")


def split_into_bar_chunks(
    part: m21.stream.Part,
    n_bars: int,
) -> Iterator[m21.stream.Stream]:
    """Группировать Measure'ы по n_bars. Pickup (paddingLeft > 0 или number == 0) — drop.
    Хвост короче n_bars — drop.
    """
    measures = list(part.getElementsByClass(m21.stream.Measure))
    full_bars = [
        m for m in measures
        if not (getattr(m, "paddingLeft", 0) > 0 or m.number == 0)
    ]
    for start in range(0, len(full_bars) - n_bars + 1, n_bars):
        chunk = m21.stream.Stream()
        for measure in full_bars[start:start + n_bars]:
            chunk.append(measure)
        yield chunk


def walk_generated_chunk_files(
    slug_dir: Path,
    model: str,
    samples_per_theme: int,
    active_themes: list[str],
    suffix: str,
) -> Iterator[Path]:
    """Walk по themes/<theme>/<model>/sample_<i>/gen_chunk_<j>.<suffix>.

    Несуществующие sample_dir игнорируются. Файлы возвращаются в
    отсортированном порядке (детерминированный walk).
    """
    glob_pattern = f"gen_chunk_*{suffix}"
    for theme in active_themes:
        for i in range(samples_per_theme):
            sample_dir = slug_dir / "themes" / theme / model / f"sample_{i}"
            if not sample_dir.is_dir():
                continue
            for path in sorted(sample_dir.glob(glob_pattern)):
                yield path
