"""Тесты human-renderer: формирование строк final_human.csv из master rows."""
from __future__ import annotations

from final_table.human import format_cell, render_human_rows
from final_table.loader import MasterRow
from final_table.registry import Group


def test_format_cell_group_a():
    """mean = X ± Y \\n median = Z (p25–p75), 3 знака после запятой."""
    row = MasterRow(
        model="cmt", metric="chord_tone_ratio",
        mean=0.477, std=0.109, median=0.473,
        p25=0.417, p75=0.544, min=0.169, max=0.748,
    )
    cell = format_cell(row, Group.A)
    assert cell == "mean = 0.477 ± 0.109\nmedian = 0.473 (0.417–0.544)"


def test_format_cell_group_b():
    row = MasterRow(model="cmt", metric="mgeval_pc", kl=0.891, oa=0.367)
    cell = format_cell(row, Group.B)
    assert cell == "KL = 0.891\nOA = 0.367"


def test_format_cell_group_c():
    row = MasterRow(model="cmt", metric="bar_rhythm_jsd", value=0.638)
    cell = format_cell(row, Group.C)
    assert cell == "0.638"


def test_format_cell_group_d():
    """LCS — той же сигнатуры что A."""
    row = MasterRow(
        model="cmt", metric="plagiarism_lcs",
        mean=6.588, std=1.5, median=6.0, p25=5.0, p75=8.0, min=1.0, max=13.0,
    )
    cell = format_cell(row, Group.D)
    assert cell == "mean = 6.588 ± 1.500\nmedian = 6.000 (5.000–8.000)"


def test_render_human_rows_shape():
    """Pivot из master rows → каждая запись словарь с metric + 3 моделями."""
    master = [
        MasterRow(model="cmt", metric="bar_rhythm_jsd", value=0.638),
        MasterRow(model="mingus", metric="bar_rhythm_jsd", value=0.652),
        MasterRow(model="bebopnet", metric="bar_rhythm_jsd", value=0.658),
    ]
    rows = render_human_rows(master)
    bjsd = next(r for r in rows if r["metric"] == "bar_rhythm_jsd")
    assert bjsd["cmt"] == "0.638"
    assert bjsd["mingus"] == "0.652"
    assert bjsd["bebopnet"] == "0.658"
