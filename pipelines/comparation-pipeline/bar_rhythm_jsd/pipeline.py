"""Bar-rhythm-JSD pipeline.

Принимает корпусы Measure'ов real и per-model gen; считает паттерны и
Jensen-Shannon divergence на парах (real, gen-of-model).
"""
from __future__ import annotations

from collections import Counter

import music21 as m21
from scipy.spatial.distance import jensenshannon

from .bar_pattern import extract_bar_pattern


def compute_jsd(real_patterns: list[str], gen_patterns: list[str]) -> float:
    """JSD на двух categorical распределениях.

    scipy.spatial.distance.jensenshannon возвращает sqrt(JSD) (Jensen-Shannon
    distance); квадрат даёт собственно JSD в нац. логарифмах. Поддержки p и q
    выравниваются на union; нули заполняются 0.0.
    """
    counter_real = Counter(real_patterns)
    counter_gen = Counter(gen_patterns)
    support = sorted(set(counter_real) | set(counter_gen))
    n_real = sum(counter_real.values())
    n_gen = sum(counter_gen.values())
    if n_real == 0 or n_gen == 0:
        raise ValueError("empty corpus: cannot compute JSD")
    p = [counter_real.get(s, 0) / n_real for s in support]
    q = [counter_gen.get(s, 0) / n_gen for s in support]
    return float(jensenshannon(p, q) ** 2)


def compute_bar_rhythm_jsd(
    real_measures: list[m21.stream.Measure],
    gen_measures_by_model: dict[str, list[m21.stream.Measure]],
) -> list[dict]:
    """Главная точка входа: вернуть list[dict] (по строке на модель).

    Поля строки: model, jsd, n_real_bars, n_gen_bars, n_unique_real,
    n_unique_gen, n_unique_union.
    """
    real_patterns = [extract_bar_pattern(m) for m in real_measures]
    real_set = set(real_patterns)

    rows: list[dict] = []
    for model, gen_ms in gen_measures_by_model.items():
        gen_patterns = [extract_bar_pattern(m) for m in gen_ms]
        gen_set = set(gen_patterns)
        rows.append({
            "model": model,
            "jsd": compute_jsd(real_patterns, gen_patterns),
            "n_real_bars": len(real_patterns),
            "n_gen_bars": len(gen_patterns),
            "n_unique_real": len(real_set),
            "n_unique_gen": len(gen_set),
            "n_unique_union": len(real_set | gen_set),
        })
    return rows
