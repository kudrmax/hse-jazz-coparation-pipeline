"""Чтение существующих _metrics/*.csv в нормализованные MasterRow.

Каждый loader-функция отвечает за один источник. Возвращают list[MasterRow] —
по одной записи на (model, metric), уже с каноническим именем метрики из registry.
Метрики не в registry — тихо игнорируются (например, удалённые pitch_range, contour_sim).
Отсутствующие в источнике метрики из registry детектятся позже в builder'е (fail-fast).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .registry import FINAL_METRICS, Group, MetricDef


@dataclass
class MasterRow:
    """Одна строка master-таблицы.

    Конкретные поля заполняются в зависимости от группы метрики (см. registry):
    - A, D: mean, std, median, p25, p75, min, max
    - B: kl, oa
    - C: value
    """
    model: str
    metric: str
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    min: float | None = None
    max: float | None = None
    kl: float | None = None
    oa: float | None = None
    value: float | None = None


def _registry_by_source_key(source_csv: str) -> dict[str, MetricDef]:
    """Карта 'имя в источнике' → MetricDef для long-формата."""
    return {
        m.source_key: m
        for m in FINAL_METRICS
        if m.source_csv == source_csv and m.source_key is not None
    }


def _parse_float_or_none(s: str) -> float | None:
    if s == "" or s is None:
        return None
    return float(s)


def load_aggregates_rows(metrics_dir: Path) -> list[MasterRow]:
    """Прочитать aggregates.csv → MasterRow для метрик группы A.

    Группа A: заполняем mean/std/median/p25/p75/min/max. Колонки n_themes игнорируем.
    Метрики, не находящиеся в registry (например, удалённые pitch_range / contour_sim) —
    тихо игнорируем. Маппинг source_key → canonical name берём из registry.
    """
    path = metrics_dir / "aggregates.csv"
    by_key = _registry_by_source_key("aggregates.csv")

    out: list[MasterRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            src_metric = r["metric"]
            mdef = by_key.get(src_metric)
            if mdef is None:
                continue
            assert mdef.group == Group.A, (
                f"aggregates.csv loader получил метрику {mdef.name} "
                f"группы {mdef.group}, ожидалась A"
            )
            out.append(MasterRow(
                model=r["model"],
                metric=mdef.name,
                mean=_parse_float_or_none(r["mean"]),
                std=_parse_float_or_none(r["std"]),
                median=_parse_float_or_none(r["median"]),
                p25=_parse_float_or_none(r["p25"]),
                p75=_parse_float_or_none(r["p75"]),
                min=_parse_float_or_none(r["min"]),
                max=_parse_float_or_none(r["max"]),
            ))
    return out


def load_mgeval_rows(metrics_dir: Path) -> list[MasterRow]:
    """Прочитать mgeval.csv → MasterRow для метрик группы B.

    Группа B: заполняем kl, oa. n_real_pieces/n_gen_pieces игнорируем.
    Маппинг feature-name (например, 'total_used_pitch') → canonical 'mgeval_pc' — из registry.
    """
    path = metrics_dir / "mgeval.csv"
    by_key = _registry_by_source_key("mgeval.csv")

    out: list[MasterRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            src_feature = r["feature"]
            mdef = by_key.get(src_feature)
            if mdef is None:
                continue
            assert mdef.group == Group.B, (
                f"mgeval.csv loader получил метрику {mdef.name} "
                f"группы {mdef.group}, ожидалась B"
            )
            out.append(MasterRow(
                model=r["model"],
                metric=mdef.name,
                kl=_parse_float_or_none(r["kl"]),
                oa=_parse_float_or_none(r["oa"]),
            ))
    return out


def load_bar_rhythm_jsd_rows(metrics_dir: Path) -> list[MasterRow]:
    """Прочитать bar_rhythm_jsd.csv → MasterRow для метрики bar_rhythm_jsd (группа C).

    Wide-формат: одна строка на модель, поле 'jsd' → value.
    Counts (n_*) игнорируем.
    """
    path = metrics_dir / "bar_rhythm_jsd.csv"
    out: list[MasterRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out.append(MasterRow(
                model=r["model"],
                metric="bar_rhythm_jsd",
                value=_parse_float_or_none(r["jsd"]),
            ))
    return out


def load_plagiarism_rows(metrics_dir: Path) -> list[MasterRow]:
    """Прочитать plagiarism.csv → MasterRow для 4 метрик: ngram_n3/n4/n5 (C) + lcs (D).

    Wide-формат: одна строка на модель, метрики закодированы именами колонок.
    Из каждой входной строки разворачиваем в 4 MasterRow.
    Counts (n_*) игнорируем.
    """
    path = metrics_dir / "plagiarism.csv"
    out: list[MasterRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            model = r["model"]
            for n in (3, 4, 5):
                out.append(MasterRow(
                    model=model,
                    metric=f"plagiarism_ngram_n{n}",
                    value=_parse_float_or_none(r[f"ngram_overlap_n{n}"]),
                ))
            out.append(MasterRow(
                model=model,
                metric="plagiarism_lcs",
                mean=_parse_float_or_none(r["lcs_max_mean"]),
                std=_parse_float_or_none(r["lcs_max_std"]),
                median=_parse_float_or_none(r["lcs_max_median"]),
                p25=_parse_float_or_none(r["lcs_max_p25"]),
                p75=_parse_float_or_none(r["lcs_max_p75"]),
                min=_parse_float_or_none(r["lcs_max_min"]),
                max=_parse_float_or_none(r["lcs_max_max"]),
            ))
    return out
