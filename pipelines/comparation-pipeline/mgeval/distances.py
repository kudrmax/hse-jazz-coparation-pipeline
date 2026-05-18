"""Port of mgeval/utils.py — Yang & Lerch (2018) reference.

1-to-1 numeric equivalence с original. Только Py3 syntax fix +
выброшен sklearn-import (нужен только для mode='EMD'/'KL' в c_dist,
которые нам не нужны).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats


def c_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances от одного A до всех в B.

    Reference: mgeval/utils.py c_dist(mode='None').
    """
    out = np.zeros(len(B))
    for i in range(len(B)):
        out[i] = np.linalg.norm(A - B[i])
    return out


def kl_dist(A: np.ndarray, B: np.ndarray, num_sample: int = 1000) -> float:
    """KL(pdf_A || pdf_B), 1-to-1 reference.

    pdf_A сэмплируется на linspace(min(A), max(A), 1000),
    pdf_B на linspace(min(B), max(B), 1000) — РАЗНЫЕ сетки.
    stats.entropy нормализует оба вектора к sum=1 внутри.
    """
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return float(stats.entropy(pdf_A(sample_A), pdf_B(sample_B)))


def overlap_area(A: np.ndarray, B: np.ndarray) -> float:
    """Area of intersection между двумя KDE-PDF, 1-to-1 reference."""
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    lo = min(np.min(A), np.min(B))
    hi = max(np.max(A), np.max(B))
    # NOTE: pdf_A(x) и pdf_B(x) возвращают shape (1,) ndarray, не scalar.
    # scipy.integrate.quad требует scalar; numpy 2.x запрещает float() на
    # 1d-array → используем .item(). Семантика reference не меняется
    # (там Py2/старший numpy неявно конвертил).
    area, _ = integrate.quad(
        lambda x: min(pdf_A(x).item(), pdf_B(x).item()),
        lo, hi,
    )
    return float(area)
