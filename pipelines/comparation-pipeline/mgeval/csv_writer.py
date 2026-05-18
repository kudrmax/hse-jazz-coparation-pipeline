"""Атомарная запись MGEval-rows в CSV."""
from __future__ import annotations

import csv
import os
from pathlib import Path


def write_mgeval_csv(rows: list[dict], path: Path) -> None:
    """Записать rows в CSV. Атомарно: tmp-file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["feature", "model", "kl", "oa", "n_real_pieces", "n_gen_pieces"]
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "kl": f"{row['kl']:.6f}",
                "oa": f"{row['oa']:.6f}",
            })
    os.replace(tmp, path)
