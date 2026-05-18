"""Общие хелперы для step1_baseline.

- `load_full_solos`: 40 test-соло → list[(name, PrettyMIDI)] без чанкинга
  (для подшага 1a).
- `load_chunks_8bar`: те же соло, нарезаны на непересекающиеся 8-bar окна
  → list[PrettyMIDI] (для подшага 1b).
- `aggregate_rows`: 5×N → 9 строк агрегата (KL/OA mean/std/min/max).
- `write_result_csv`: запись CSV в требуемом ТЗ формате (точность .6f).

Используется наша MGEval-реализация из `mgeval/` (никаких сторонних импортов).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import music21 as m21
import numpy as np
import pretty_midi

# Корень пакета comparation-pipeline (там лежит mgeval/).
_COMP_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
if str(_COMP_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMP_PIPELINE_ROOT))

from mgeval.corpus_loader import (  # noqa: E402
    _extract_melody_part,
    _note_count,
    _score_chunk_to_pretty_midi,
    _split_into_bar_chunks,
)
from mgeval.features import FEATURES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
SPLIT_JSON = REPO_ROOT / "pipelines" / "training-pipeline" / "wjazzd_split.json"
XML_DIR = REPO_ROOT / "models" / "MINGUS" / "A_preprocessData" / "data" / "xml"

SEEDS = [42, 123, 7, 999, 2024]


def _full_bars(part: m21.stream.Part) -> list[m21.stream.Measure]:
    """Все Measure'ы, кроме pickup (paddingLeft>0 либо number==0)."""
    measures = list(part.getElementsByClass(m21.stream.Measure))
    return [m for m in measures if not (getattr(m, "paddingLeft", 0) > 0 or m.number == 0)]


def load_full_solos(
    split_json_path: Path = SPLIT_JSON,
    xml_dir: Path = XML_DIR,
) -> list[tuple[str, pretty_midi.PrettyMIDI]]:
    """40 test-соло → [(name, PrettyMIDI)]. Pickup-measures отбрасываются.
    Дегенеративные (0 нот) выкидываются.
    """
    test_names = json.loads(Path(split_json_path).read_text())["test"]
    out: list[tuple[str, pretty_midi.PrettyMIDI]] = []
    for name in test_names:
        xml_path = Path(xml_dir) / f"{name}.xml"
        score = m21.converter.parse(str(xml_path))
        melody_part = _extract_melody_part(score)
        bars = _full_bars(melody_part)
        if not bars:
            continue
        chunk = m21.stream.Stream()
        for measure in bars:
            chunk.append(measure)
        pm = _score_chunk_to_pretty_midi(chunk)
        if _note_count(pm) > 0:
            out.append((name, pm))
    return out


def load_chunks_8bar(
    split_json_path: Path = SPLIT_JSON,
    xml_dir: Path = XML_DIR,
    chunk_bars: int = 8,
) -> tuple[list[pretty_midi.PrettyMIDI], dict[str, int]]:
    """40 test-соло → list[PrettyMIDI] непересекающихся `chunk_bars`-окон.
    Хвосты <chunk_bars отбрасываются. Возвращает также {solo_name: n_chunks}.
    """
    test_names = json.loads(Path(split_json_path).read_text())["test"]
    chunks: list[pretty_midi.PrettyMIDI] = []
    counts: dict[str, int] = {}
    for name in test_names:
        xml_path = Path(xml_dir) / f"{name}.xml"
        score = m21.converter.parse(str(xml_path))
        melody_part = _extract_melody_part(score)
        n_added = 0
        for chunk_stream in _split_into_bar_chunks(melody_part, chunk_bars):
            pm = _score_chunk_to_pretty_midi(chunk_stream)
            if _note_count(pm) > 0:
                chunks.append(pm)
                n_added += 1
        counts[name] = n_added
    return chunks, counts


def aggregate_rows(
    per_trial_rows: list[list[dict]],
) -> dict[str, dict[str, list[float]]]:
    """[trial][row] → {feature: {"kl": [...], "oa": [...]}} (по 5 trial'ов на feature)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"kl": [], "oa": []})
    for trial in per_trial_rows:
        for row in trial:
            acc[row["feature"]]["kl"].append(row["kl"])
            acc[row["feature"]]["oa"].append(row["oa"])
    return dict(acc)


def write_result_csv(
    csv_path: Path,
    agg: dict[str, dict[str, list[float]]],
) -> None:
    """CSV строго по ТЗ:
    feature, kl_mean, kl_std, kl_min, kl_max, oa_mean, oa_std, oa_min, oa_max, n_trials
    Порядок строк = порядок в FEATURES registry (9 фич). Точность 6 знаков.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature",
            "kl_mean", "kl_std", "kl_min", "kl_max",
            "oa_mean", "oa_std", "oa_min", "oa_max",
            "n_trials",
        ])
        for feat_name in FEATURES.keys():
            kls = np.asarray(agg[feat_name]["kl"], dtype=float)
            oas = np.asarray(agg[feat_name]["oa"], dtype=float)
            w.writerow([
                feat_name,
                f"{kls.mean():.6f}", f"{kls.std():.6f}", f"{kls.min():.6f}", f"{kls.max():.6f}",
                f"{oas.mean():.6f}", f"{oas.std():.6f}", f"{oas.min():.6f}", f"{oas.max():.6f}",
                len(kls),
            ])
