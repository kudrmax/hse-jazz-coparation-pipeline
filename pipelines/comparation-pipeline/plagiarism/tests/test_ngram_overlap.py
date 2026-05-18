"""Тесты n-gram overlap (multiset-pool, B2)."""
from __future__ import annotations

from collections import Counter

import pytest

from plagiarism.ngram_overlap import (
    compute_ngram_overlap,
    ngrams,
    train_ngram_set,
)


def test_ngrams_basic():
    assert ngrams([1, 2, 3, 4], 3) == Counter([(1, 2, 3), (2, 3, 4)])


def test_ngrams_too_short():
    assert ngrams([1, 2], 3) == Counter()


def test_ngrams_repeats():
    # На [1,2,1,2,1,2] длины 6 → 4 трёхграммы (позиции 0..3):
    # (1,2,1), (2,1,2), (1,2,1), (2,1,2) → Counter({(1,2,1): 2, (2,1,2): 2})
    assert ngrams([1, 2, 1, 2, 1, 2], 3) == Counter([
        (1, 2, 1), (2, 1, 2), (1, 2, 1), (2, 1, 2),
    ])


def test_train_ngram_set_union():
    train = [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
    ]
    result = train_ngram_set(train, n=3)
    # solo1 → {(1,2,3), (2,3,4)}; solo2 → {(3,4,5), (4,5,6)}
    assert result == {(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)}


def test_train_ngram_set_skips_short():
    """Соло короче n не вносит n-граммы (но не падает)."""
    train = [[1, 2], [1, 2, 3]]  # первое слишком короткое
    assert train_ngram_set(train, n=3) == {(1, 2, 3)}


def test_overlap_identical_gen_full_match():
    """gen-pool совпадает с train_set → overlap = 1.0."""
    train_set = {(1, 2, 3), (2, 3, 4)}
    gen_chunks = [[1, 2, 3, 4]]
    overlap, total = compute_ngram_overlap(gen_chunks, train_set, n=3)
    assert overlap == 1.0
    assert total == 2


def test_overlap_disjoint_zero():
    train_set = {(1, 2, 3)}
    gen_chunks = [[10, 20, 30, 40]]
    overlap, total = compute_ngram_overlap(gen_chunks, train_set, n=3)
    assert overlap == 0.0
    assert total == 2


def test_overlap_multiset_weighting():
    """B2: если n-грамма повторяется в gen 3 раза, count = 3 в числитель."""
    train_set = {(1, 2, 3)}
    # n-грамм в [1,2,3,1,2,3,1,2,3,1,2,3] (длины 12):
    # (1,2,3) на позициях 0,3,6,9 → 4 раза
    # (2,3,1) на 1, 4, 7 → 3 раза
    # (3,1,2) на 2, 5, 8 → 3 раза
    # ИТОГО 10 n-грамм, из них 4 в train_set
    gen_chunks = [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]]
    overlap, total = compute_ngram_overlap(gen_chunks, train_set, n=3)
    assert total == 10
    assert overlap == 4 / 10


def test_overlap_pool_across_multiple_chunks():
    """Pool суммирует по всем chunks (не per-chunk averaging)."""
    train_set = {(1, 2, 3)}
    gen_chunks = [
        [1, 2, 3],         # 1 n-грамма (1,2,3) — в train
        [10, 20, 30, 40],  # 2 n-граммы (10,20,30), (20,30,40) — НЕ в train
    ]
    overlap, total = compute_ngram_overlap(gen_chunks, train_set, n=3)
    assert total == 3       # 1 + 2
    assert overlap == 1 / 3 # 1 совпадение из 3


def test_overlap_empty_pool_raises():
    """Пустой gen_multiset → ValueError (все chunks слишком короткие)."""
    train_set = {(1, 2, 3)}
    gen_chunks = [[1, 2], []]
    with pytest.raises(ValueError):
        compute_ngram_overlap(gen_chunks, train_set, n=3)
