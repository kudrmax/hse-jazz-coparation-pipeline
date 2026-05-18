"""PlagiarismPipeline: adapter над plagiarism/ модулями.

Особенность: _load_real_corpus игнорирует chunk_bars (train не нарезается).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from comparation_config import ComparationConfig
from metric_pipelines.base import BaseCorpusMetricPipeline

REPO_ROOT = Path(__file__).resolve().parents[3]


class PlagiarismPipeline(BaseCorpusMetricPipeline):
    name = "plagiarism"
    csv_filename = "plagiarism.csv"

    def _load_real_corpus(self, cfg: ComparationConfig) -> Any:
        from plagiarism.corpus_loader import load_train_corpus_intervals
        return load_train_corpus_intervals(
            split_json_path=REPO_ROOT / "pipelines/training-pipeline/wjazzd_split.json",
            xml_dir=REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml",
        )

    def _load_gen_corpus(
        self, slug_dir: Path, model: str,
        samples_per_theme: int, active_themes: list[str],
    ) -> Any:
        from plagiarism.corpus_loader import iter_generated_corpus_intervals
        return iter_generated_corpus_intervals(
            slug_dir, model, samples_per_theme, active_themes,
        )

    def _compute(self, real: Any, gen_by_model: dict[str, Any]) -> list[dict]:
        from plagiarism.pipeline import compute_plagiarism
        return compute_plagiarism(real, gen_by_model)

    def _write_csv(self, rows: list[dict], out_path: Path) -> None:
        from plagiarism.csv_writer import write_plagiarism_csv
        write_plagiarism_csv(rows, out_path)

    def _log_real_size(self, real: Any) -> None:
        print(f"{self.name}: train corpus = {len(real)} pieces")

    def _log_gen_size(self, model: str, gen: Any) -> None:
        print(f"{self.name}: gen[{model}] = {len(gen)} chunks")
