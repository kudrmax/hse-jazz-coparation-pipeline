"""Longest Common Substring (непрерывная подстрока) через numpy DP.

Используется rolling buffer на 2 строки → память O(min(L_a, L_b)).
Inner loop векторизован по строке матрицы: для текущего a[i] построить
маску match вдоль b и обновить curr[1:] за один np.where.
"""
from __future__ import annotations

import numpy as np


def lcs_length(a: list[int], b: list[int]) -> int:
    """Длина самой длинной непрерывной общей подстроки между a и b.

    Возвращает 0 если a или b пустой.
    """
    if not a or not b:
        return 0
    arr_a = np.asarray(a, dtype=np.int32)
    arr_b = np.asarray(b, dtype=np.int32)
    L_b = arr_b.shape[0]
    prev = np.zeros(L_b + 1, dtype=np.int32)
    curr = np.zeros(L_b + 1, dtype=np.int32)
    best = 0
    for i in range(arr_a.shape[0]):
        match = (arr_b == arr_a[i])
        curr[1:] = np.where(match, prev[:-1] + 1, 0)
        cur_max = int(curr.max())
        if cur_max > best:
            best = cur_max
        prev, curr = curr, prev
    return best
