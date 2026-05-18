"""MGEval pipeline: features → pairwise distances → KDE → KL+OA.

На вход — два списка PrettyMIDI-кусков (real, gen-by-model).
На выход — list[dict], по одной строке на (feature × model).
"""
from __future__ import annotations

import numpy as np
import pretty_midi

from .distances import c_dist, kl_dist, overlap_area
from .features import FEATURES


def _extract_features(
    corpus: list[pretty_midi.PrettyMIDI],
    feature_fn,
) -> np.ndarray:
    """Применить feature_fn к каждому куску, выкинуть None'ы, вернуть stack."""
    vectors = []
    for pm in corpus:
        v = feature_fn(pm)
        if v is None:
            continue
        v = np.asarray(v).flatten().astype(float)
        vectors.append(v)
    if not vectors:
        raise ValueError("All feature extractions returned None — empty corpus?")
    return np.stack(vectors, axis=0)


def _intra_distances(X: np.ndarray) -> np.ndarray:
    """Все попарные расстояния (i, j), i != j. Reference LOOCV-pattern."""
    n = len(X)
    if n < 2:
        raise ValueError(f"intra requires >=2 samples, got {n}")
    out = np.zeros((n, n - 1))
    for i in range(n):
        mask = np.arange(n) != i
        out[i] = c_dist(X[i], X[mask])
    return out.flatten()


def _inter_distances(X_real: np.ndarray, X_gen: np.ndarray) -> np.ndarray:
    """Outer-product real × gen. Reference."""
    n_real = len(X_real)
    n_gen = len(X_gen)
    if n_real < 1 or n_gen < 1:
        raise ValueError(f"inter requires both >=1, got {n_real}, {n_gen}")
    out = np.zeros((n_real, n_gen))
    for i in range(n_real):
        out[i] = c_dist(X_real[i], X_gen)
    return out.flatten()


def compute_mgeval(
    real_corpus: list[pretty_midi.PrettyMIDI],
    gen_corpora_by_model: dict[str, list[pretty_midi.PrettyMIDI]],
) -> list[dict]:
    """Главная точка входа: вернуть list[dict] (по строке на feature × model)."""
    rows: list[dict] = []
    n_real = len(real_corpus)
    for feature_name, feature_fn in FEATURES.items():
        X_real = _extract_features(real_corpus, feature_fn)
        intra_real = _intra_distances(X_real)
        for model, gen_corpus in gen_corpora_by_model.items():
            X_gen = _extract_features(gen_corpus, feature_fn)
            inter = _inter_distances(X_real, X_gen)
            rows.append({
                "feature": feature_name,
                "model": model,
                "kl": kl_dist(intra_real, inter),
                "oa": overlap_area(intra_real, inter),
                "n_real_pieces": n_real,
                "n_gen_pieces": len(gen_corpus),
            })
    return rows
