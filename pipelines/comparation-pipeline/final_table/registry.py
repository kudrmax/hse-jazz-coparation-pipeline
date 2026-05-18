"""Канонический registry метрик финальной сводной таблицы.

SSOT для:
- порядка метрик в final.csv / final_human.csv (один к одному с paper/metric_final_list.md);
- группы (A/B/C/D) → определяет какие колонки заполнены;
- маппинга в источник (какой CSV, какое поле).

Внешние модули (loader, builder) импортируют FINAL_METRICS и опираются на этот
порядок. Менять состав — только синхронно с paper/metric_final_list.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Group(Enum):
    A = auto()  # per-theme distribution (mean/std/median/p25/p75/min/max)
    B = auto()  # MGEval: kl + oa
    C = auto()  # corpus-level scalar (value)
    D = auto()  # distribution over chunks (как A)


@dataclass(frozen=True)
class MetricDef:
    """Описание одной метрики в финальной таблице.

    name — каноническое имя для колонки `metric` в final.csv.
    group — определяет заполняемые колонки.
    source_csv — имя CSV-файла в _metrics/.
    source_key — для long-формата (aggregates.csv, mgeval.csv): значение поля
                 ('metric' или 'feature'), которое идентифицирует строку.
                 None — для wide-формата, где метрика закодирована именами колонок.
    """
    name: str
    group: Group
    source_csv: str
    source_key: str | None


FINAL_METRICS: tuple[MetricDef, ...] = (
    # Группа B — MGEval (9)
    MetricDef("mgeval_pc",    Group.B, "mgeval.csv", "total_used_pitch"),
    MetricDef("mgeval_pr",    Group.B, "mgeval.csv", "pitch_range"),
    MetricDef("mgeval_pi",    Group.B, "mgeval.csv", "avg_pitch_interval"),
    MetricDef("mgeval_nc",    Group.B, "mgeval.csv", "total_used_note"),
    MetricDef("mgeval_ioi",   Group.B, "mgeval.csv", "avg_ioi"),
    MetricDef("mgeval_pch",   Group.B, "mgeval.csv", "total_pitch_class_histogram"),
    MetricDef("mgeval_pctm",  Group.B, "mgeval.csv", "pitch_class_transition_matrix"),
    MetricDef("mgeval_nlh",   Group.B, "mgeval.csv", "note_length_hist"),
    MetricDef("mgeval_nltm",  Group.B, "mgeval.csv", "note_length_transition_matrix"),
    # Группа A — per-theme distribution (10), порядок по metric_final_list.md
    MetricDef("scale_match",            Group.A, "aggregates.csv", "scale_match"),
    MetricDef("scale_match_per_time",   Group.A, "aggregates.csv", "scale_match_per_time"),
    MetricDef("chord_tone_ratio",       Group.A, "aggregates.csv", "ctr"),
    MetricDef("ctr_first_beat",         Group.A, "aggregates.csv", "ctr_first_beat"),
    MetricDef("chord_match_per_time",   Group.A, "aggregates.csv", "chord_match_per_time"),
    MetricDef("bar_rhythm_jsd",         Group.C, "bar_rhythm_jsd.csv", None),
    MetricDef("note_density",           Group.A, "aggregates.csv", "note_density"),
    MetricDef("pitch_entropy",          Group.A, "aggregates.csv", "pitch_entropy"),
    MetricDef("theme_ngram_overlap_n3", Group.A, "aggregates.csv", "ngram_3_overlap"),
    MetricDef("theme_ngram_overlap_n4", Group.A, "aggregates.csv", "ngram_4_overlap"),
    MetricDef("theme_ngram_overlap_n5", Group.A, "aggregates.csv", "ngram_5_overlap"),
    # Группа C — plagiarism n-gram (3)
    MetricDef("plagiarism_ngram_n3", Group.C, "plagiarism.csv", None),
    MetricDef("plagiarism_ngram_n4", Group.C, "plagiarism.csv", None),
    MetricDef("plagiarism_ngram_n5", Group.C, "plagiarism.csv", None),
    # Группа D — LCS (1)
    MetricDef("plagiarism_lcs", Group.D, "plagiarism.csv", None),
)
