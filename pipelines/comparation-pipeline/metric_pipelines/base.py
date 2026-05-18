"""BaseCorpusMetricPipeline: ABC для corpus-level метрик (mgeval/bar_rhythm/plagiarism).

Template Method: `run()` фиксирует sequence load real → load gen per model →
compute → write CSV. Наследник реализует hooks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from comparation_config import ComparationConfig
from manifest import Manifest
from model_names import MODEL_NAMES as _MODELS


class BaseCorpusMetricPipeline(ABC):
    """Полиморфная стратегия для одной corpus-level метрики (один CSV)."""

    name: str          # class-level: "mgeval" | "bar_rhythm_jsd" | "plagiarism"
    csv_filename: str  # class-level: "mgeval.csv" | "bar_rhythm_jsd.csv" | ...

    # === Template Method ===
    def run(
        self, slug_dir: Path, manifest: Manifest, cfg: ComparationConfig,
    ) -> None:
        real = self._load_real_corpus(cfg)
        self._log_real_size(real)

        active_themes = manifest.active_themes()
        gen_by_model: dict[str, Any] = {}
        for model in _MODELS:
            gen_by_model[model] = self._load_gen_corpus(
                slug_dir, model, manifest.samples_per_theme, active_themes,
            )
            self._log_gen_size(model, gen_by_model[model])

        rows = self._compute(real, gen_by_model)
        out_path = slug_dir / "_metrics" / self.csv_filename
        self._write_csv(rows, out_path)
        print(f"{self.csv_filename}: {len(rows)} rows → {out_path}")

    # === Hooks (наследник обязан реализовать) ===
    @abstractmethod
    def _load_real_corpus(self, cfg: ComparationConfig) -> Any: ...

    @abstractmethod
    def _load_gen_corpus(
        self, slug_dir: Path, model: str,
        samples_per_theme: int, active_themes: list[str],
    ) -> Any: ...

    @abstractmethod
    def _compute(self, real: Any, gen_by_model: dict[str, Any]) -> list[dict]: ...

    @abstractmethod
    def _write_csv(self, rows: list[dict], out_path: Path) -> None: ...

    # === Опциональные hooks ===
    def _log_real_size(self, real: Any) -> None:
        print(f"{self.name}: real corpus = {len(real)}")

    def _log_gen_size(self, model: str, gen: Any) -> None:
        print(f"{self.name}: gen[{model}] = {len(gen)}")
