"""Тесты атомарных writer'ов для final.csv и final_human.csv."""
from __future__ import annotations

import csv

from final_table.io import (
    FINAL_FIELDNAMES,
    HUMAN_FIELDNAMES,
    write_final_csv,
    write_final_human_csv,
)
from final_table.loader import MasterRow


def test_write_final_csv_creates_file_with_header(tmp_path):
    out = tmp_path / "final.csv"
    rows = [
        MasterRow(model="cmt", metric="chord_tone_ratio",
                  mean=0.477, std=0.109, median=0.473,
                  p25=0.417, p75=0.544, min=0.169, max=0.748),
        MasterRow(model="cmt", metric="mgeval_pc", kl=0.891, oa=0.367),
        MasterRow(model="cmt", metric="bar_rhythm_jsd", value=0.638),
    ]
    write_final_csv(rows, out)

    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FINAL_FIELDNAMES
        loaded = list(reader)
    assert len(loaded) == 3
    assert loaded[0]["metric"] == "chord_tone_ratio"
    assert loaded[0]["mean"] == "0.477000"
    assert loaded[0]["kl"] == ""


def test_write_final_csv_atomic_no_tmp_left(tmp_path):
    out = tmp_path / "final.csv"
    write_final_csv([MasterRow(model="cmt", metric="bar_rhythm_jsd", value=0.5)], out)
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_final_csv_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "_metrics" / "final.csv"
    write_final_csv([MasterRow(model="cmt", metric="bar_rhythm_jsd", value=0.5)], out)
    assert out.exists()


def test_write_final_human_csv_basic(tmp_path):
    out = tmp_path / "final_human.csv"
    rows = [
        {"metric": "chord_tone_ratio",
         "cmt": "0.477 ± 0.109\n0.473 (0.417–0.544)",
         "mingus": "0.392 ± 0.077\n0.381 (0.330–0.438)",
         "bebopnet": "0.352 ± 0.065\n0.344 (0.311–0.378)"},
        {"metric": "bar_rhythm_jsd",
         "cmt": "0.638", "mingus": "0.652", "bebopnet": "0.658"},
    ]
    write_final_human_csv(rows, out)
    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == HUMAN_FIELDNAMES
        loaded = list(reader)
    assert "\n" in loaded[0]["cmt"]
    assert loaded[0]["cmt"].startswith("0.477")
    assert loaded[1]["mingus"] == "0.652"


def test_write_final_human_csv_atomic_no_tmp(tmp_path):
    out = tmp_path / "final_human.csv"
    write_final_human_csv(
        [{"metric": "x", "cmt": "1", "mingus": "2", "bebopnet": "3"}], out,
    )
    assert list(tmp_path.glob("*.tmp")) == []
