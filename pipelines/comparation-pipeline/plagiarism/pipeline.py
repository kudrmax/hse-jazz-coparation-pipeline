"""Plagiarism pipeline — главный compute_plagiarism.

Принимает train_corpus (list[list[int]] интервалов, без chunking) и
gen_corpora_by_model (dict[model -> list[list[int]] per chunk]). Считает
multiset-pool n-gram overlap для каждого n ∈ N_VALUES и LCS per chunk
(max-LCS по train) с агрегацией mean/median/min/max.

Возвращает list[dict] — по одной строке на модель.
"""
from __future__ import annotations

import sys

import numpy as np

from .lcs import lcs_length
from .ngram_overlap import compute_ngram_overlap, train_ngram_set

N_VALUES: tuple[int, ...] = (3, 4, 5)
_LCS_PROGRESS_EVERY = 100


def compute_plagiarism(
    train_corpus: list[list[int]],
    gen_corpora_by_model: dict[str, list[list[int]]],
) -> list[dict]:
    """Главная точка входа. См. spec §Phase 1 → pipeline.py."""
    if not train_corpus:
        raise ValueError("Empty train corpus")
    n_train = len(train_corpus)

    # Train n-gram sets — построить один раз для всех моделей.
    print(
        f"plagiarism: building train n-gram sets for n in {N_VALUES}...",
        file=sys.stderr, flush=True,
    )
    train_sets: dict[int, set[tuple[int, ...]]] = {
        n: train_ngram_set(train_corpus, n) for n in N_VALUES
    }
    sizes_str = ", ".join(f"n{n}:{len(train_sets[n])}" for n in N_VALUES)
    print(f"plagiarism: train sets size — {sizes_str}", file=sys.stderr, flush=True)

    rows: list[dict] = []
    for model, gen_chunks in gen_corpora_by_model.items():
        n_gen_chunks = len(gen_chunks)
        if n_gen_chunks == 0:
            raise ValueError(f"Empty gen corpus for model {model}")

        # N-gram overlap (multiset-pool по всем chunks модели).
        overlaps: dict[int, float] = {}
        pool_sizes: dict[int, int] = {}
        for n in N_VALUES:
            overlap, total = compute_ngram_overlap(gen_chunks, train_sets[n], n)
            overlaps[n] = overlap
            pool_sizes[n] = total

        # LCS per chunk → max по train → агрегация по chunks.
        lcs_max_per_chunk: list[int] = []
        non_empty = [c for c in gen_chunks if c]
        total_non_empty = len(non_empty)
        for i, chunk in enumerate(non_empty):
            max_lcs = max(lcs_length(chunk, t) for t in train_corpus)
            lcs_max_per_chunk.append(max_lcs)
            if (i + 1) % _LCS_PROGRESS_EVERY == 0:
                print(
                    f"plagiarism: lcs[{model}] chunk {i + 1}/{total_non_empty}",
                    file=sys.stderr, flush=True,
                )

        if not lcs_max_per_chunk:
            raise ValueError(f"No non-empty gen chunks for LCS in model {model}")

        arr = np.asarray(lcs_max_per_chunk, dtype=np.float64)
        rows.append({
            "model": model,
            "ngram_overlap_n3": overlaps[3],
            "ngram_overlap_n4": overlaps[4],
            "ngram_overlap_n5": overlaps[5],
            "lcs_max_mean": float(arr.mean()),
            "lcs_max_std": float(arr.std(ddof=0)),
            "lcs_max_median": float(np.median(arr)),
            "lcs_max_p25": float(np.percentile(arr, 25)),
            "lcs_max_p75": float(np.percentile(arr, 75)),
            "lcs_max_min": int(arr.min()),
            "lcs_max_max": int(arr.max()),
            "n_gen_chunks": n_gen_chunks,
            "n_gen_chunks_lcs": len(lcs_max_per_chunk),
            "n_gen_ngrams_n3": pool_sizes[3],
            "n_gen_ngrams_n4": pool_sizes[4],
            "n_gen_ngrams_n5": pool_sizes[5],
            "n_train_pieces": n_train,
        })
    return rows
