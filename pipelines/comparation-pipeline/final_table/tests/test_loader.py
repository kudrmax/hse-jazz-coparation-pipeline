"""Тесты loader-функций — чтение существующих CSV в master rows."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from final_table.loader import (
    MasterRow,
    load_aggregates_rows,
    load_bar_rhythm_jsd_rows,
    load_mgeval_rows,
    load_plagiarism_rows,
)


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_aggregates_rows_basic(tmp_path):
    """Чтение aggregates.csv → MasterRow для каждой метрики группы A × моделей."""
    agg = tmp_path / "_metrics" / "aggregates.csv"
    _write_csv(
        agg,
        header=["model", "metric", "n_themes", "mean", "std", "median",
                "p25", "p75", "min", "max"],
        rows=[
            {"model": "cmt", "metric": "ctr", "n_themes": 99,
             "mean": "0.477", "std": "0.109", "median": "0.473",
             "p25": "0.417", "p75": "0.544", "min": "0.169", "max": "0.748"},
            {"model": "bebopnet", "metric": "ctr", "n_themes": 98,
             "mean": "0.352", "std": "0.065", "median": "0.344",
             "p25": "0.311", "p75": "0.378", "min": "0.244", "max": "0.627"},
        ],
    )

    rows = load_aggregates_rows(tmp_path / "_metrics")

    ctr_cmt = next(r for r in rows if r.metric == "chord_tone_ratio" and r.model == "cmt")
    assert ctr_cmt.mean == pytest.approx(0.477)
    assert ctr_cmt.std == pytest.approx(0.109)
    assert ctr_cmt.median == pytest.approx(0.473)
    assert ctr_cmt.p25 == pytest.approx(0.417)
    assert ctr_cmt.p75 == pytest.approx(0.544)
    assert ctr_cmt.min == pytest.approx(0.169)
    assert ctr_cmt.max == pytest.approx(0.748)
    assert ctr_cmt.kl is None and ctr_cmt.oa is None and ctr_cmt.value is None


def test_load_aggregates_unknown_metric_ignored(tmp_path):
    """Метрика, отсутствующая в registry, тихо игнорируется (например, удалённая pitch_range)."""
    agg = tmp_path / "_metrics" / "aggregates.csv"
    _write_csv(
        agg,
        header=["model", "metric", "n_themes", "mean", "std", "median",
                "p25", "p75", "min", "max"],
        rows=[
            {"model": "cmt", "metric": "pitch_range", "n_themes": 99,
             "mean": "11.0", "std": "2.0", "median": "11.0",
             "p25": "9.0", "p75": "13.0", "min": "7.0", "max": "16.0"},
        ],
    )
    rows = load_aggregates_rows(tmp_path / "_metrics")
    assert rows == []


def test_load_mgeval_rows_basic(tmp_path):
    mg = tmp_path / "_metrics" / "mgeval.csv"
    _write_csv(
        mg,
        header=["feature", "model", "kl", "oa", "n_real_pieces", "n_gen_pieces"],
        rows=[
            {"feature": "total_used_pitch", "model": "cmt",
             "kl": "0.891", "oa": "0.367", "n_real_pieces": "337", "n_gen_pieces": "700"},
            {"feature": "total_used_pitch", "model": "bebopnet",
             "kl": "0.009", "oa": "0.854", "n_real_pieces": "337", "n_gen_pieces": "479"},
            {"feature": "pitch_range", "model": "cmt",
             "kl": "1.017", "oa": "0.302", "n_real_pieces": "337", "n_gen_pieces": "700"},
        ],
    )
    rows = load_mgeval_rows(tmp_path / "_metrics")

    pc_cmt = next(r for r in rows if r.metric == "mgeval_pc" and r.model == "cmt")
    assert pc_cmt.kl == pytest.approx(0.891)
    assert pc_cmt.oa == pytest.approx(0.367)
    assert pc_cmt.mean is None and pc_cmt.value is None

    pr_cmt = next(r for r in rows if r.metric == "mgeval_pr" and r.model == "cmt")
    assert pr_cmt.kl == pytest.approx(1.017)
    assert pr_cmt.oa == pytest.approx(0.302)


def test_load_bar_rhythm_jsd_rows(tmp_path):
    br = tmp_path / "_metrics" / "bar_rhythm_jsd.csv"
    _write_csv(
        br,
        header=["model", "jsd", "n_real_bars", "n_gen_bars",
                "n_unique_real", "n_unique_gen", "n_unique_union"],
        rows=[
            {"model": "cmt", "jsd": "0.637", "n_real_bars": "2696",
             "n_gen_bars": "8324", "n_unique_real": "2022",
             "n_unique_gen": "1561", "n_unique_union": "3553"},
            {"model": "bebopnet", "jsd": "0.658", "n_real_bars": "2696",
             "n_gen_bars": "7648", "n_unique_real": "2022",
             "n_unique_gen": "2644", "n_unique_union": "4625"},
        ],
    )
    rows = load_bar_rhythm_jsd_rows(tmp_path / "_metrics")
    cmt = next(r for r in rows if r.model == "cmt")
    assert cmt.metric == "bar_rhythm_jsd"
    assert cmt.value == pytest.approx(0.637)
    assert cmt.kl is None and cmt.mean is None


def test_load_plagiarism_rows(tmp_path):
    plag = tmp_path / "_metrics" / "plagiarism.csv"
    _write_csv(
        plag,
        header=[
            "model",
            "ngram_overlap_n3", "ngram_overlap_n4", "ngram_overlap_n5",
            "lcs_max_mean", "lcs_max_std", "lcs_max_median",
            "lcs_max_p25", "lcs_max_p75",
            "lcs_max_min", "lcs_max_max",
            "n_gen_chunks", "n_gen_chunks_lcs",
            "n_gen_ngrams_n3", "n_gen_ngrams_n4", "n_gen_ngrams_n5",
            "n_train_pieces",
        ],
        rows=[{
            "model": "cmt",
            "ngram_overlap_n3": "0.994880",
            "ngram_overlap_n4": "0.903423",
            "ngram_overlap_n5": "0.541678",
            "lcs_max_mean": "6.588489", "lcs_max_std": "1.5",
            "lcs_max_median": "6.0",
            "lcs_max_p25": "5.0", "lcs_max_p75": "8.0",
            "lcs_max_min": "1", "lcs_max_max": "13",
            "n_gen_chunks": "700", "n_gen_chunks_lcs": "695",
            "n_gen_ngrams_n3": "19142",
            "n_gen_ngrams_n4": "18462",
            "n_gen_ngrams_n5": "17791",
            "n_train_pieces": "340",
        }],
    )
    rows = load_plagiarism_rows(tmp_path / "_metrics")

    metrics = {r.metric for r in rows}
    assert metrics == {
        "plagiarism_ngram_n3", "plagiarism_ngram_n4",
        "plagiarism_ngram_n5", "plagiarism_lcs",
    }

    ng3 = next(r for r in rows if r.metric == "plagiarism_ngram_n3")
    assert ng3.value == pytest.approx(0.994880)
    assert ng3.mean is None and ng3.kl is None

    lcs = next(r for r in rows if r.metric == "plagiarism_lcs")
    assert lcs.mean == pytest.approx(6.588489)
    assert lcs.std == pytest.approx(1.5)
    assert lcs.median == pytest.approx(6.0)
    assert lcs.p25 == pytest.approx(5.0)
    assert lcs.p75 == pytest.approx(8.0)
    assert lcs.min == pytest.approx(1.0)
    assert lcs.max == pytest.approx(13.0)
    assert lcs.value is None
