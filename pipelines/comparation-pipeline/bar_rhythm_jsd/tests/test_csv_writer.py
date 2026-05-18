"""Тесты csv_writer."""
from __future__ import annotations

import csv

from bar_rhythm_jsd.csv_writer import write_bar_rhythm_jsd_csv


def test_write_writes_header_and_rows(tmp_path):
    rows = [
        {"model": "cmt", "jsd": 0.123456,
         "n_real_bars": 2696, "n_gen_bars": 1584,
         "n_unique_real": 421, "n_unique_gen": 89, "n_unique_union": 480},
        {"model": "mingus", "jsd": 0.0,
         "n_real_bars": 2696, "n_gen_bars": 1584,
         "n_unique_real": 421, "n_unique_gen": 421, "n_unique_union": 421},
    ]
    out = tmp_path / "bar_rhythm_jsd.csv"
    write_bar_rhythm_jsd_csv(rows, out)

    assert out.exists()
    with out.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[0]["model"] == "cmt"
    assert float(reader[0]["jsd"]) == 0.123456
    assert int(reader[0]["n_unique_union"]) == 480


def test_write_atomic_no_partial_on_failure(tmp_path):
    """tmp-файла после успешной записи не должно остаться."""
    out = tmp_path / "bar_rhythm_jsd.csv"
    write_bar_rhythm_jsd_csv([{
        "model": "cmt", "jsd": 0.0,
        "n_real_bars": 1, "n_gen_bars": 1,
        "n_unique_real": 1, "n_unique_gen": 1, "n_unique_union": 1,
    }], out)
    assert out.exists()
    assert not (out.with_suffix(out.suffix + ".tmp")).exists()
