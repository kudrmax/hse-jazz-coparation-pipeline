"""Атомарная запись plagiarism rows в CSV."""
from __future__ import annotations

import csv
import os
from pathlib import Path

FIELDNAMES = [
    "model",
    "ngram_overlap_n3", "ngram_overlap_n4", "ngram_overlap_n5",
    "lcs_max_mean", "lcs_max_std", "lcs_max_median",
    "lcs_max_p25", "lcs_max_p75",
    "lcs_max_min", "lcs_max_max",
    "n_gen_chunks", "n_gen_chunks_lcs",
    "n_gen_ngrams_n3", "n_gen_ngrams_n4", "n_gen_ngrams_n5",
    "n_train_pieces",
]

_FLOAT_FIELDS = (
    "ngram_overlap_n3", "ngram_overlap_n4", "ngram_overlap_n5",
    "lcs_max_mean", "lcs_max_std", "lcs_max_median",
    "lcs_max_p25", "lcs_max_p75",
)


def write_plagiarism_csv(rows: list[dict], path: Path) -> None:
    """Записать rows в CSV атомарно: tmp-file + os.replace.

    Float-колонки (overlaps + lcs_max_mean/median) форматируются как {val:.6f}.
    Int-колонки (n_*, lcs_max_min/max) — стандартный str().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            for k in _FLOAT_FIELDS:
                out_row[k] = f"{float(row[k]):.6f}"
            writer.writerow(out_row)
    os.replace(tmp, path)
