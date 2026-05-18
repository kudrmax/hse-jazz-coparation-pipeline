"""Тесты атомарной записи plagiarism CSV."""
from __future__ import annotations

import csv

from plagiarism.csv_writer import FIELDNAMES, write_plagiarism_csv


def _sample_row(model: str) -> dict:
    return {
        "model": model,
        "ngram_overlap_n3": 0.1234567,
        "ngram_overlap_n4": 0.0,
        "ngram_overlap_n5": 1.0,
        "lcs_max_mean": 2.3333333,
        "lcs_max_std": 1.4142136,
        "lcs_max_median": 2.0,
        "lcs_max_p25": 1.5,
        "lcs_max_p75": 3.0,
        "lcs_max_min": 1,
        "lcs_max_max": 5,
        "n_gen_chunks": 100,
        "n_gen_chunks_lcs": 98,
        "n_gen_ngrams_n3": 500,
        "n_gen_ngrams_n4": 450,
        "n_gen_ngrams_n5": 400,
        "n_train_pieces": 344,
    }


def test_write_creates_file_with_header_and_rows(tmp_path):
    out = tmp_path / "plagiarism.csv"
    rows = [_sample_row("cmt"), _sample_row("mingus")]
    write_plagiarism_csv(rows, out)

    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDNAMES
        loaded = list(reader)
    assert len(loaded) == 2
    assert loaded[0]["model"] == "cmt"
    assert loaded[1]["model"] == "mingus"


def test_float_columns_formatted_6_digits(tmp_path):
    out = tmp_path / "plagiarism.csv"
    write_plagiarism_csv([_sample_row("cmt")], out)

    with out.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["ngram_overlap_n3"] == "0.123457"  # 6 знаков
    assert row["lcs_max_mean"] == "2.333333"
    assert row["lcs_max_std"] == "1.414214"
    assert row["lcs_max_p25"] == "1.500000"
    assert row["lcs_max_p75"] == "3.000000"
    assert row["lcs_max_min"] == "1"  # int без форматирования
    assert row["n_train_pieces"] == "344"


def test_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "dir" / "plagiarism.csv"
    write_plagiarism_csv([_sample_row("cmt")], out)
    assert out.exists()


def test_atomic_no_tmp_left_behind(tmp_path):
    out = tmp_path / "plagiarism.csv"
    write_plagiarism_csv([_sample_row("cmt")], out)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
