"""Тесты LCS (Longest Common Substring) — включая sanity checks из ТЗ."""
from __future__ import annotations

import random

from plagiarism.lcs import lcs_length


def test_lcs_identical():
    a = [1, 2, 3, 4, 5]
    assert lcs_length(a, a) == 5


def test_lcs_disjoint_returns_zero():
    a = [1, 2, 3]
    b = [10, 20, 30]
    assert lcs_length(a, b) == 0


def test_lcs_empty_first():
    assert lcs_length([], [1, 2, 3]) == 0


def test_lcs_empty_second():
    assert lcs_length([1, 2, 3], []) == 0


def test_lcs_both_empty():
    assert lcs_length([], []) == 0


def test_lcs_known_substring():
    # Общая подстрока [2, 3, 4] длины 3
    a = [1, 2, 3, 4, 5]
    b = [9, 2, 3, 4, 8]
    assert lcs_length(a, b) == 3


def test_lcs_at_head():
    a = [1, 2, 3, 99, 99]
    b = [1, 2, 3, 88, 88]
    assert lcs_length(a, b) == 3


def test_lcs_at_tail():
    a = [99, 99, 1, 2, 3]
    b = [88, 88, 1, 2, 3]
    assert lcs_length(a, b) == 3


def test_lcs_single_element_match():
    a = [1, 2, 3]
    b = [99, 2, 99]
    assert lcs_length(a, b) == 1


def test_lcs_no_subsequence_gaps():
    """Substring (continuous), не Subsequence (with gaps).
    Общая subsequence-with-gaps была бы [1,2,3] длины 3,
    но непрерывной подстроки длиннее 1 нет.
    """
    a = [1, 99, 2, 99, 3]
    b = [1, 2, 3]
    assert lcs_length(a, b) == 1


def test_lcs_random_sanity():
    """На случайных int-последовательностях из большого алфавита
    LCS должен быть короткий (typically 1-3).
    """
    rng = random.Random(42)
    a = [rng.randint(-50, 50) for _ in range(200)]
    b = [rng.randint(-50, 50) for _ in range(200)]
    result = lcs_length(a, b)
    # На алфавите ~100 и длине 200 случайные совпадения 3+ редки
    assert 0 <= result <= 5


def test_lcs_negative_intervals():
    """Интервалы могут быть отрицательными — должно работать без проблем."""
    a = [-5, 3, -7, 12]
    b = [99, -5, 3, -7, 88]
    assert lcs_length(a, b) == 3
