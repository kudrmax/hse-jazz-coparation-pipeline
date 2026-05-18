"""MetricsPhase: координатор фазы 2 (расчёт метрик и сохранение CSV).

PerSegmentMetricsRunner — главный цикл themes × models × samples × chunks
с расчётом per-segment.csv → aggregates.csv → significance.csv.

MetricsPhase — фасад фазы: запустить per-segment runner, затем corpus-level
pipeline'ы.
"""
from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from comparation_config import ComparationConfig
from final_table import FinalTableStep
from manifest import Manifest
from model_names import MODEL_NAMES as _MODELS

if TYPE_CHECKING:
    from metric_pipelines.base import BaseCorpusMetricPipeline

# metrics/ — пакет с per-segment метриками, импорт через sys.path
_COMP_ROOT = Path(__file__).resolve().parents[1]
if str(_COMP_ROOT / "metrics") not in sys.path:
    sys.path.insert(0, str(_COMP_ROOT / "metrics"))


class PerSegmentMetricsRunner:
    """Считает per_segment.csv → aggregates.csv → significance.csv."""

    def __init__(self, metrics: list) -> None:
        self.metrics = metrics
        self.metric_names = [m.name for m in metrics]

    def run(self, slug_dir: Path, manifest: Manifest, chunk_bars: int) -> None:
        metrics_dir = slug_dir / "_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        rows = self._compute_per_segment(slug_dir, manifest, chunk_bars)
        per_segment_path = metrics_dir / "per_segment.csv"
        self._write_per_segment(rows, per_segment_path)
        print(f"per_segment.csv: {len(rows)} rows → {per_segment_path}")

        per_theme = self._aggregate_per_theme(per_segment_path)

        agg_path = metrics_dir / "aggregates.csv"
        self._write_aggregates(per_theme, agg_path)

        sig_path = metrics_dir / "significance.csv"
        self._write_significance(per_theme, sig_path)

    def _compute_per_segment(
        self, slug_dir: Path, manifest: Manifest, chunk_bars: int,
    ) -> list[dict]:
        import music21 as m21
        import pretty_midi
        from base import SegmentContext  # type: ignore[import-not-found]

        rows: list[dict] = []
        for theme_name in manifest.active_themes():
            theme_dir = slug_dir / "themes" / theme_name
            chunks_dir = theme_dir / "theme_chunks"
            theme_chunk_files = sorted(chunks_dir.glob("chunk_*.musicxml"))

            theme_chunk_scores: list[m21.stream.Score | None] = []
            for ch in theme_chunk_files:
                try:
                    theme_chunk_scores.append(m21.converter.parse(str(ch)))
                except Exception as e:
                    print(f"  warn parsing {ch}: {e}")
                    theme_chunk_scores.append(None)

            for model in _MODELS:
                for idx in range(manifest.samples_per_theme):
                    sample_dir = theme_dir / model / f"sample_{idx}"
                    if not sample_dir.is_dir():
                        continue
                    for j, theme_score in enumerate(theme_chunk_scores):
                        chunk_mid = sample_dir / f"gen_chunk_{j}.mid"
                        if not chunk_mid.exists():
                            continue
                        pm = pretty_midi.PrettyMIDI(str(chunk_mid))
                        ctx = SegmentContext(
                            segment=pm,
                            chord_context=theme_score,
                            comparison_melody=theme_score,
                            bars=chunk_bars,
                        )
                        row: dict[str, object] = {
                            "theme": theme_name, "model": model,
                            "sample_idx": idx, "chunk_idx": j,
                            "n_notes": sum(len(i.notes) for i in pm.instruments),
                        }
                        for metric in self.metrics:
                            try:
                                v = metric.compute(ctx)
                            except Exception as e:
                                print(f"  metric {metric.name} failed on "
                                      f"{theme_name}/{model}/{idx}/{j}: {e}")
                                v = None
                            row[metric.name] = "" if v is None else f"{v:.6f}"
                        rows.append(row)
        return rows

    def _write_per_segment(self, rows: list[dict], path: Path) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "theme", "model", "sample_idx", "chunk_idx", "n_notes",
                    *self.metric_names,
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _aggregate_per_theme(self, per_segment_path: Path):
        import pandas as pd

        df = pd.read_csv(per_segment_path)
        if df.empty or not all(col in df.columns for col in self.metric_names):
            return pd.DataFrame(columns=["model", "theme"] + self.metric_names)
        return df.groupby(["model", "theme"])[self.metric_names].mean(
            numeric_only=True,
        ).reset_index()

    def _write_aggregates(self, per_theme, path: Path) -> None:
        import pandas as pd

        agg_rows: list[dict] = []
        for model, g in per_theme.groupby("model"):
            for col in self.metric_names:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                agg_rows.append({
                    "model": model, "metric": col,
                    "n_themes": int(vals.shape[0]),
                    "mean": float(vals.mean()) if len(vals) else None,
                    "std": float(vals.std()) if len(vals) > 1 else None,
                    "median": float(vals.median()) if len(vals) else None,
                    "p25": float(vals.quantile(0.25)) if len(vals) else None,
                    "p75": float(vals.quantile(0.75)) if len(vals) else None,
                    "min": float(vals.min()) if len(vals) else None,
                    "max": float(vals.max()) if len(vals) else None,
                })
        pd.DataFrame(agg_rows).to_csv(path, index=False)
        print(f"aggregates.csv: per-theme aggregation (B), {len(agg_rows)} rows → {path}")

    def _write_significance(self, per_theme, path: Path) -> None:
        import pandas as pd
        from scipy import stats

        sig_rows: list[dict] = []
        for col in self.metric_names:
            for ma, mb in combinations(_MODELS, 2):
                a = pd.to_numeric(
                    per_theme.loc[per_theme["model"] == ma, col],
                    errors="coerce",
                ).dropna()
                b = pd.to_numeric(
                    per_theme.loc[per_theme["model"] == mb, col],
                    errors="coerce",
                ).dropna()
                if len(a) < 5 or len(b) < 5:
                    u, p = None, None
                else:
                    try:
                        res = stats.mannwhitneyu(a, b, alternative="two-sided")
                        u, p = float(res.statistic), float(res.pvalue)
                    except ValueError:
                        u, p = None, None
                sig_rows.append({
                    "metric": col, "model_a": ma, "model_b": mb,
                    "n_a": len(a), "n_b": len(b),
                    "u_statistic": u, "p_value": p,
                    "significant_at_0.05": (p is not None and p < 0.05),
                })
        pd.DataFrame(sig_rows).to_csv(path, index=False)
        print(f"significance.csv: {len(sig_rows)} rows → {path}")


class MetricsPhase:
    """Фасад фазы 2: per-segment runner + corpus-level pipelines."""

    def __init__(
        self,
        per_segment_runner: PerSegmentMetricsRunner,
        corpus_pipelines: list["BaseCorpusMetricPipeline"],
    ) -> None:
        self.per_segment_runner = per_segment_runner
        self.corpus_pipelines = corpus_pipelines

    def run(self, slug_dir: Path, cfg: ComparationConfig) -> None:
        manifest_path = slug_dir / "manifest.json"
        if not manifest_path.exists():
            raise MetricsPhaseError(f"manifest.json not found: {manifest_path}")
        manifest = Manifest.load(manifest_path)

        (slug_dir / "_metrics").mkdir(parents=True, exist_ok=True)

        self.per_segment_runner.run(slug_dir, manifest, cfg.segmentation.chunk_bars)
        for pipeline in self.corpus_pipelines:
            pipeline.run(slug_dir, manifest, cfg)
        FinalTableStep(slug_dir).run()


class MetricsPhaseError(Exception):
    """Raised on missing manifest / другие orchestration issues."""
