"""Backfill xml для existing comparation-pipeline outputs.

Для каждого gen_chunk_<j>.mid в slug-папке дописывает рядом
gen_chunk_<j>.musicxml и gen_chunk_<j>_with_chords.musicxml.

Запуск (под pipeline venv, т.к. MidiToMusicxmlConverter живёт в нём):
  pipelines/generation-pipeline/.venv/bin/python \\
      pipelines/comparation-pipeline/backfill_musicxml.py \\
      --slug full-cleared-2samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import music21 as m21
import pretty_midi
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"
GEN_ROOT = REPO_ROOT / "pipelines/generation-pipeline"
sys.path.insert(0, str(COMP_ROOT))
sys.path.insert(0, str(GEN_ROOT))

from models.base.midi_to_musicxml import MidiToMusicxmlConverter  # type: ignore  # noqa: E402


BackfillResult = tuple[str, str]
# statuses per slot: "written" | "skipped" | "midi_load_failed" | "missing_theme_chunk"


def extract_chord_symbols(
    theme_chunk_xml: Path,
) -> list[tuple[float, m21.harmony.ChordSymbol]]:
    """Распарсить theme_chunks/chunk_<j>.musicxml и вытащить ChordSymbol'ы
    с их offsets от начала chunk'а (в quarterLength).

    Использует score.flatten() — offset во flattened stream совпадает с
    глобальным offset'ом от начала piece.
    """
    score = m21.converter.parse(str(theme_chunk_xml))
    flat = score.flatten()
    result: list[tuple[float, m21.harmony.ChordSymbol]] = []
    for cs in flat.getElementsByClass(m21.harmony.ChordSymbol):
        result.append((float(cs.offset), cs))
    return result


def backfill_one_chunk(
    mid_path: Path,
    theme_chunk_xml: Path,
    chunk_bars: int,
) -> BackfillResult:
    """Дописать .musicxml и _with_chords.musicxml рядом с .mid файлом.

    Идемпотентно: если xml уже существует — пропуск этого артефакта.
    Если theme_chunk_xml отсутствует — мелодийный xml всё равно пишется,
    _with_chords помечается как missing_theme_chunk.
    Если .mid битый — оба возвращают "midi_load_failed".
    """
    melody_path = mid_path.with_suffix(".musicxml")
    with_chords_path = mid_path.parent / f"{mid_path.stem}_with_chords.musicxml"

    melody_status: str
    with_chords_status: str

    try:
        pm = pretty_midi.PrettyMIDI(str(mid_path))
    except Exception as e:  # noqa: BLE001
        print(f"midi load failed: {mid_path}: {e}", file=sys.stderr)
        return ("midi_load_failed", "midi_load_failed")

    # melody-only xml
    if melody_path.exists():
        melody_status = "skipped"
    else:
        score = MidiToMusicxmlConverter.to_melody_only(pm)
        score.write("musicxml", fp=str(melody_path))
        melody_status = "written"

    # with-chords xml
    if with_chords_path.exists():
        with_chords_status = "skipped"
    elif not theme_chunk_xml.exists():
        with_chords_status = "missing_theme_chunk"
    else:
        chord_syms = extract_chord_symbols(theme_chunk_xml)
        score_wc = MidiToMusicxmlConverter.to_with_chords(
            pm, chord_syms, input_bars=chunk_bars, output_bars=0,
        )
        score_wc.write("musicxml", fp=str(with_chords_path))
        with_chords_status = "written"

    return (melody_status, with_chords_status)


from model_names import MODEL_NAMES as _MODELS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill_musicxml.py")
    parser.add_argument("--slug", required=True)
    args = parser.parse_args(argv)

    slug_dir = COMP_ROOT / "outputs" / args.slug
    if not slug_dir.is_dir():
        print(f"slug dir not found: {slug_dir}", file=sys.stderr)
        return 2

    config_snapshot = slug_dir / "config.snapshot.yaml"
    if not config_snapshot.exists():
        print(f"config.snapshot.yaml missing: {config_snapshot}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(config_snapshot.read_text())
    chunk_bars = int(cfg["segmentation"]["chunk_bars"])

    counts: dict[str, int] = {
        "melody_written": 0, "with_chords_written": 0,
        "skipped": 0, "midi_load_failed": 0, "missing_theme_chunk": 0,
    }
    themes_root = slug_dir / "themes"
    if not themes_root.is_dir():
        print(f"no themes/ in {slug_dir}", file=sys.stderr)
        return 2

    for theme_dir in sorted(themes_root.iterdir()):
        if not theme_dir.is_dir():
            continue
        for model in _MODELS:
            model_dir = theme_dir / model
            if not model_dir.is_dir():
                continue
            for sample_dir in sorted(model_dir.glob("sample_*")):
                for mid_path in sorted(sample_dir.glob("gen_chunk_*.mid")):
                    j = mid_path.stem.removeprefix("gen_chunk_")
                    theme_chunk_xml = theme_dir / "theme_chunks" / f"chunk_{j}.musicxml"
                    melody_status, with_chords_status = backfill_one_chunk(
                        mid_path, theme_chunk_xml, chunk_bars,
                    )
                    for status in (melody_status, with_chords_status):
                        if status == "skipped":
                            counts["skipped"] += 1
                        elif status == "midi_load_failed":
                            counts["midi_load_failed"] += 1
                        elif status == "missing_theme_chunk":
                            counts["missing_theme_chunk"] += 1
                    if melody_status == "written":
                        counts["melody_written"] += 1
                    if with_chords_status == "written":
                        counts["with_chords_written"] += 1

    print(f"backfill done: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
