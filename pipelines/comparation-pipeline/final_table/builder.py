"""FinalTableBuilder — собирает 72 строки master-таблицы из 4 входных CSV.

Порядок строк: 24 метрики из FINAL_METRICS × 3 модели из MODEL_NAMES (cmt, mingus, bebopnet).
Если для метрики нет данных в источнике хотя бы для одной модели — ValueError (fail-fast).
"""
from __future__ import annotations

from pathlib import Path

from model_names import MODEL_NAMES  # type: ignore[import-not-found]

from .loader import (
    MasterRow,
    load_aggregates_rows,
    load_bar_rhythm_jsd_rows,
    load_mgeval_rows,
    load_plagiarism_rows,
)
from .registry import FINAL_METRICS


class FinalTableBuilder:
    def __init__(self, metrics_dir: Path) -> None:
        self.metrics_dir = metrics_dir

    def build(self) -> list[MasterRow]:
        index = self._load_index()
        out: list[MasterRow] = []
        for mdef in FINAL_METRICS:
            for model in MODEL_NAMES:
                key = (mdef.name, model)
                row = index.get(key)
                if row is None:
                    raise ValueError(
                        f"final_table: метрика {mdef.name!r} для модели {model!r} "
                        f"не найдена в источнике {mdef.source_csv}. "
                        f"Проверь что соответствующий пайплайн отработал."
                    )
                out.append(row)
        return out

    def _load_index(self) -> dict[tuple[str, str], MasterRow]:
        all_rows: list[MasterRow] = []
        all_rows.extend(load_aggregates_rows(self.metrics_dir))
        all_rows.extend(load_mgeval_rows(self.metrics_dir))
        all_rows.extend(load_bar_rhythm_jsd_rows(self.metrics_dir))
        all_rows.extend(load_plagiarism_rows(self.metrics_dir))
        return {(r.metric, r.model): r for r in all_rows}
