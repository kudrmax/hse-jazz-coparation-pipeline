"""Atomic writers для final.csv (machine-readable) и final_human.csv (для ВКР).

Оба пишут через tmp-file + os.replace, чтобы потребитель не увидел частично
записанный CSV даже если процесс упадёт посередине.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from .loader import MasterRow


FINAL_FIELDNAMES: list[str] = [
    "model", "metric",
    "mean", "std", "median", "p25", "p75", "min", "max",
    "kl", "oa", "value",
]

_FLOAT_COLS: tuple[str, ...] = (
    "mean", "std", "median", "p25", "p75", "min", "max",
    "kl", "oa", "value",
)


def _format_float(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}"


def write_final_csv(rows: list[MasterRow], path: Path) -> None:
    """Записать master rows в final.csv атомарно.

    NULL значения → пустая строка. Float колонки → {v:.6f}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            out_row: dict[str, str] = {"model": r.model, "metric": r.metric}
            for col in _FLOAT_COLS:
                out_row[col] = _format_float(getattr(r, col))
            writer.writerow(out_row)
    os.replace(tmp, path)


HUMAN_FIELDNAMES: list[str] = ["metric", "cmt", "mingus", "bebopnet"]


def write_final_human_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Записать human-таблицу в CSV атомарно.

    Ячейки могут содержать '\\n' (двухстрочные). Стандартный csv-модуль
    оборачивает их в quotes, корректно читается Excel/Numbers/LibreOffice.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)
