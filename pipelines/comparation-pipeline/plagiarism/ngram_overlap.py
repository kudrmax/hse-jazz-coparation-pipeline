"""Multiset-pool n-gram overlap (вариант B2 из ТЗ).

`gen_multiset` строится как Counter всех n-грамм всех chunks одной модели
(не per-chunk usrednenie). Overlap = доля count'ов, попадающих в train_set.

Семантика: «доля сгенерированных мелодических event'ов, совпадающих с train»
(если модель повторяет заученную фразу N раз, это считается N copy events).
"""
from __future__ import annotations

from collections import Counter


def ngrams(seq: list[int], n: int) -> Counter:
    """Counter всех n-грамм длины n из seq. Пуст если len(seq) < n."""
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def train_ngram_set(
    train_corpus: list[list[int]],
    n: int,
) -> set[tuple[int, ...]]:
    """Union всех n-грамм всех train-соло как set (быстрый O(1) lookup).

    Соло короче n вносят 0 n-грамм (не падают).
    """
    result: set[tuple[int, ...]] = set()
    for solo in train_corpus:
        if len(solo) < n:
            continue
        for i in range(len(solo) - n + 1):
            result.add(tuple(solo[i:i + n]))
    return result


def compute_ngram_overlap(
    gen_chunks_intervals: list[list[int]],
    train_set: set[tuple[int, ...]],
    n: int,
) -> tuple[float, int]:
    """Multiset-pool overlap.

    1. Сложить все n-граммы всех chunks в один Counter.
    2. Overlap = Σ count(g) для g ∈ train_set / Σ count(g) всех g.

    Возвращает (overlap, total_count_in_pool).
    Если pool пуст → ValueError.
    """
    pool: Counter = Counter()
    for chunk in gen_chunks_intervals:
        pool.update(ngrams(chunk, n))
    total = sum(pool.values())
    if total == 0:
        raise ValueError(f"Empty gen multiset-pool for n={n}")
    matched = sum(count for g, count in pool.items() if g in train_set)
    return matched / total, total
