"""Тесты канонического registry метрик финальной таблицы."""
from __future__ import annotations

from final_table.registry import FINAL_METRICS, Group, MetricDef


def test_final_metrics_count_24():
    """24 метрики в финальной таблице по spec'е."""
    assert len(FINAL_METRICS) == 24


def test_final_metrics_unique_names():
    names = [m.name for m in FINAL_METRICS]
    assert len(names) == len(set(names))


def test_final_metrics_order_starts_with_mgeval():
    """Порядок строго по paper/metric_final_list.md — MGEval идёт первым."""
    assert FINAL_METRICS[0].name == "mgeval_pc"
    assert FINAL_METRICS[8].name == "mgeval_nltm"


def test_group_distribution():
    """Распределение метрик по группам: A=10, B=9, C=4, D=1."""
    counts: dict[Group, int] = {g: 0 for g in Group}
    for m in FINAL_METRICS:
        counts[m.group] += 1
    assert counts[Group.A] == 10
    assert counts[Group.B] == 9
    assert counts[Group.C] == 4
    assert counts[Group.D] == 1


def test_mgeval_metrics_in_group_b():
    for m in FINAL_METRICS:
        if m.name.startswith("mgeval_"):
            assert m.group == Group.B


def test_plagiarism_lcs_in_group_d():
    lcs = [m for m in FINAL_METRICS if m.name == "plagiarism_lcs"]
    assert len(lcs) == 1
    assert lcs[0].group == Group.D


def test_metric_def_has_source_mapping():
    """Каждая MetricDef знает источник (csv-файл + поле / колонка)."""
    for m in FINAL_METRICS:
        assert m.source_csv
        assert hasattr(m, "source_key")
