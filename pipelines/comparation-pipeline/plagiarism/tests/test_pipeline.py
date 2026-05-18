"""Тесты compute_plagiarism — end-to-end на synthetic корпусах."""
from __future__ import annotations

import pytest

from plagiarism.pipeline import N_VALUES, compute_plagiarism


def _row_by_model(rows: list[dict], model: str) -> dict:
    matches = [r for r in rows if r["model"] == model]
    assert len(matches) == 1
    return matches[0]


def test_compute_plagiarism_basic_shape():
    """Возвращает строку на каждую модель со всеми ожидаемыми ключами."""
    train = [[1, 2, 3, 4, 5, 6, 7, 8]]
    gen = {
        "cmt": [[1, 2, 3, 4, 5, 6]],
        "mingus": [[1, 2, 3, 4, 5, 6]],
    }
    rows = compute_plagiarism(train, gen)
    assert len(rows) == 2
    expected_keys = {
        "model",
        "ngram_overlap_n3", "ngram_overlap_n4", "ngram_overlap_n5",
        "lcs_max_mean", "lcs_max_std", "lcs_max_median",
        "lcs_max_p25", "lcs_max_p75",
        "lcs_max_min", "lcs_max_max",
        "n_gen_chunks", "n_gen_chunks_lcs",
        "n_gen_ngrams_n3", "n_gen_ngrams_n4", "n_gen_ngrams_n5",
        "n_train_pieces",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys


def test_compute_plagiarism_identical_train_gen_overlap_one():
    """Если gen точно из train → overlap = 1.0."""
    train = [[1, 2, 3, 4, 5, 6]]
    gen = {"cmt": [[1, 2, 3, 4, 5, 6]]}
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["ngram_overlap_n3"] == 1.0
    assert row["ngram_overlap_n4"] == 1.0
    assert row["ngram_overlap_n5"] == 1.0


def test_compute_plagiarism_identical_train_gen_lcs_full_length():
    """Если gen точно из train → LCS = длина."""
    train = [[1, 2, 3, 4, 5, 6]]
    gen = {"cmt": [[1, 2, 3, 4, 5, 6]]}
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["lcs_max_mean"] == 6.0
    assert row["lcs_max_max"] == 6


def test_compute_plagiarism_disjoint_overlap_zero():
    """gen disjoint с train → overlap = 0, LCS = 0."""
    train = [[1, 2, 3, 4, 5]]
    gen = {"cmt": [[100, 200, 300, 400, 500]]}
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["ngram_overlap_n3"] == 0.0
    assert row["lcs_max_mean"] == 0.0
    assert row["lcs_max_max"] == 0


def test_compute_plagiarism_symmetry_identical_gen():
    """Identical gen для двух моделей → identical всех метрик."""
    train = [[1, 2, 3, 4, 5, 6, 7]]
    gen_intervals = [[1, 2, 99, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    gen = {"cmt": gen_intervals, "mingus": gen_intervals}
    rows = compute_plagiarism(train, gen)
    cmt = _row_by_model(rows, "cmt")
    mingus = _row_by_model(rows, "mingus")
    for k in cmt:
        if k == "model":
            continue
        assert cmt[k] == mingus[k]


def test_compute_plagiarism_n_chunks_counts():
    """n_gen_chunks включает все chunks; n_gen_chunks_lcs только с ≥1 интервал."""
    train = [[1, 2, 3, 4, 5, 6, 7]]
    gen = {
        "cmt": [[1, 2, 3, 4, 5, 6], [1, 2], [], []],  # 4 chunks
    }
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["n_gen_chunks"] == 4
    # chunks с ≥1 интервалом: [1,2,3,4,5,6]→непуст, [1,2]→непуст, [], [] → пустые
    assert row["n_gen_chunks_lcs"] == 2


def test_compute_plagiarism_empty_train_raises():
    train: list[list[int]] = []
    gen = {"cmt": [[1, 2, 3]]}
    with pytest.raises(ValueError, match="train"):
        compute_plagiarism(train, gen)


def test_compute_plagiarism_no_lcs_chunks_raises():
    """Все chunks модели пустые → ValueError (multiset-pool пуст)."""
    train = [[1, 2, 3]]
    gen = {"cmt": [[], [], []]}  # все пустые
    with pytest.raises(ValueError):
        compute_plagiarism(train, gen)


def test_compute_plagiarism_n_train_pieces():
    train = [[1, 2, 3, 4, 5, 6, 7], [10, 11, 12, 13, 14, 15, 16, 17]]
    gen = {"cmt": [[1, 2, 3, 4, 5, 6]]}
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["n_train_pieces"] == 2


def test_compute_plagiarism_n_values_constant():
    """N_VALUES = (3, 4, 5) — публичная константа."""
    assert N_VALUES == (3, 4, 5)


def test_compute_plagiarism_lcs_std_p25_p75_known_values():
    """LCS-распределение из 5 фиксированных чанков → проверяем std/p25/p75."""
    # Train с 7 интервалами, gen — 5 chunks разной похожести → LCS = [6, 5, 4, 3, 2].
    train = [[1, 2, 3, 4, 5, 6, 7]]
    gen = {
        "cmt": [
            [1, 2, 3, 4, 5, 6],          # LCS = 6
            [1, 2, 3, 4, 5],              # LCS = 5
            [1, 2, 3, 4],                 # LCS = 4
            [1, 2, 3],                    # LCS = 3
            [1, 2],                       # LCS = 2
        ],
    }
    rows = compute_plagiarism(train, gen)
    row = _row_by_model(rows, "cmt")
    assert row["lcs_max_mean"] == pytest.approx(4.0)
    # std с ddof=0 (population) для [2,3,4,5,6] = sqrt(2) ≈ 1.4142
    assert row["lcs_max_std"] == pytest.approx(1.4142135623730951, rel=1e-6)
    # numpy percentile linear на [2,3,4,5,6]: p25 = 3.0, p75 = 5.0
    assert row["lcs_max_p25"] == pytest.approx(3.0)
    assert row["lcs_max_p75"] == pytest.approx(5.0)
