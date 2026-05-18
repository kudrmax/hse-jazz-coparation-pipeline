"""Атомарная запись bar-rhythm-jsd rows в CSV."""
from __future__ import annotations

import csv
import os
from pathlib import Path


FIELDNAMES = [
    "model", "jsd",
    "n_real_bars", "n_gen_bars",
    "n_unique_real", "n_unique_gen", "n_unique_union",
]


def write_bar_rhythm_jsd_csv(rows: list[dict], path: Path) -> None:
    """Записать rows в CSV. Атомарно: tmp-file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "jsd": f"{row['jsd']:.6f}",
            })
    os.replace(tmp, path)
