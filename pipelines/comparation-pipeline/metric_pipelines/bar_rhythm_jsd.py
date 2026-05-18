"""BarRhythmJsdPipeline: adapter над bar_rhythm_jsd/ модулями."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from comparation_config import ComparationConfig
from metric_pipelines.base import BaseCorpusMetricPipeline

REPO_ROOT = Path(__file__).resolve().parents[3]


class BarRhythmJsdPipeline(BaseCorpusMetricPipeline):
    name = "bar_rhythm_jsd"
    csv_filename = "bar_rhythm_jsd.csv"

    def _load_real_corpus(self, cfg: ComparationConfig) -> Any:
        from bar_rhythm_jsd.corpus_loader import load_real_corpus_measures
        return load_real_corpus_measures(
            split_json_path=REPO_ROOT / "pipelines/training-pipeline/wjazzd_split.json",
            xml_dir=REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml",
            chunk_bars=cfg.segmentation.chunk_bars,
        )

    def _load_gen_corpus(
        self, slug_dir: Path, model: str,
        samples_per_theme: int, active_themes: list[str],
    ) -> Any:
        from bar_rhythm_jsd.corpus_loader import iter_generated_corpus_measures
        return iter_generated_corpus_measures(
            slug_dir, model, samples_per_theme, active_themes,
        )

    def _compute(self, real: Any, gen_by_model: dict[str, Any]) -> list[dict]:
        from bar_rhythm_jsd.pipeline import compute_bar_rhythm_jsd
        return compute_bar_rhythm_jsd(real, gen_by_model)

    def _write_csv(self, rows: list[dict], out_path: Path) -> None:
        from bar_rhythm_jsd.csv_writer import write_bar_rhythm_jsd_csv
        write_bar_rhythm_jsd_csv(rows, out_path)

    def _log_real_size(self, real: Any) -> None:
        print(f"{self.name}: real corpus = {len(real)} bars")

    def _log_gen_size(self, model: str, gen: Any) -> None:
        print(f"{self.name}: gen[{model}] = {len(gen)} bars")
