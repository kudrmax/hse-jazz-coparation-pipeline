"""Тесты pipeline (JSD + full computation)."""
from __future__ import annotations

import math

from bar_rhythm_jsd.pipeline import compute_bar_rhythm_jsd, compute_jsd
from bar_rhythm_jsd.tests._helpers import make_measure


def test_jsd_identical_distributions_is_zero():
    p = ["A", "A", "B", "B", "C"]
    assert compute_jsd(p, list(p)) == 0.0


def test_jsd_disjoint_distributions_is_ln2():
    # Распределения с disjoint support: JSD достигает максимума ln(2)
    a = ["X"] * 100
    b = ["Y"] * 100
    jsd = compute_jsd(a, b)
    assert math.isclose(jsd, math.log(2), rel_tol=1e-6)


def test_jsd_symmetric():
    a = ["A"] * 5 + ["B"] * 3
    b = ["A"] * 2 + ["B"] * 7
    assert math.isclose(compute_jsd(a, b), compute_jsd(b, a), rel_tol=1e-10)


def test_jsd_in_bounds():
    a = ["A", "B", "C"]
    b = ["A", "B", "D"]
    jsd = compute_jsd(a, b)
    assert 0.0 <= jsd <= math.log(2) + 1e-9


def _quarter_bar():
    return make_measure([(i * 1.0, 1.0, "C4") for i in range(4)])


def _full_rest_bar():
    return make_measure([(0.0, 4.0, None)])


def test_compute_bar_rhythm_jsd_identical_corpora():
    real = [_quarter_bar() for _ in range(10)]
    gen = {"cmt": [_quarter_bar() for _ in range(10)]}
    rows = compute_bar_rhythm_jsd(real, gen)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "cmt"
    assert row["jsd"] == 0.0
    assert row["n_real_bars"] == 10
    assert row["n_gen_bars"] == 10
    assert row["n_unique_real"] == 1
    assert row["n_unique_gen"] == 1
    assert row["n_unique_union"] == 1


def test_compute_bar_rhythm_jsd_disjoint_corpora():
    real = [_quarter_bar() for _ in range(10)]
    gen = {"cmt": [_full_rest_bar() for _ in range(10)]}
    rows = compute_bar_rhythm_jsd(real, gen)
    assert len(rows) == 1
    row = rows[0]
    assert math.isclose(row["jsd"], math.log(2), rel_tol=1e-6)
    assert row["n_unique_real"] == 1
    assert row["n_unique_gen"] == 1
    assert row["n_unique_union"] == 2


def test_compute_bar_rhythm_jsd_three_models():
    real = [_quarter_bar() for _ in range(5)]
    gen = {
        "cmt": [_quarter_bar() for _ in range(5)],
        "mingus": [_full_rest_bar() for _ in range(5)],
        "bebopnet": [_quarter_bar() for _ in range(5)],
    }
    rows = compute_bar_rhythm_jsd(real, gen)
    assert {r["model"] for r in rows} == {"cmt", "mingus", "bebopnet"}
    assert sum(1 for r in rows if r["jsd"] == 0.0) == 2  # cmt и bebopnet идентичны real
