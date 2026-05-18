"""FinalTableStep — точка входа в финальный шаг metrics-фазы.

Читает 4 существующих CSV из <slug_dir>/_metrics/, собирает мастер-таблицу
по registry и пишет <slug_dir>/_metrics/final.csv + final_human.csv атомарно.
"""
from __future__ import annotations

from pathlib import Path

from .builder import FinalTableBuilder
from .human import render_human_rows
from .io import write_final_csv, write_final_human_csv


class FinalTableStep:
    def __init__(self, slug_dir: Path) -> None:
        self.slug_dir = slug_dir

    def run(self) -> None:
        metrics_dir = self.slug_dir / "_metrics"
        master_rows = FinalTableBuilder(metrics_dir).build()
        write_final_csv(master_rows, metrics_dir / "final.csv")

        human_rows = render_human_rows(master_rows)
        write_final_human_csv(human_rows, metrics_dir / "final_human.csv")
        print(
            f"final_table: wrote {len(master_rows)} master rows "
            f"and {len(human_rows)} human rows → {metrics_dir}",
        )
