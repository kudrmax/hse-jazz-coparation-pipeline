"""Тесты FinalTableBuilder — сборка master rows из 4 CSV в порядке registry."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from final_table.builder import FinalTableBuilder
from final_table.registry import FINAL_METRICS


_MODELS = ("cmt", "mingus", "bebopnet")


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _setup_full_inputs(metrics_dir: Path) -> None:
    """Записать минимально-валидные aggregates/mgeval/plagiarism/bar_rhythm_jsd."""
    agg_metric_names = [
        "scale_match", "scale_match_per_time", "ctr", "ctr_first_beat",
        "chord_match_per_time", "note_density", "pitch_entropy",
        "ngram_3_overlap", "ngram_4_overlap", "ngram_5_overlap",
    ]
    agg_rows = []
    for m in _MODELS:
        for name in agg_metric_names:
            agg_rows.append({
                "model": m, "metric": name, "n_themes": 99,
                "mean": "0.5", "std": "0.1", "median": "0.5",
                "p25": "0.4", "p75": "0.6", "min": "0.0", "max": "1.0",
            })
    _write_csv(
        metrics_dir / "aggregates.csv",
        header=["model", "metric", "n_themes", "mean", "std", "median",
                "p25", "p75", "min", "max"],
        rows=agg_rows,
    )

    mg_features = [
        "total_used_pitch", "pitch_range", "avg_pitch_interval",
        "total_used_note", "avg_ioi", "total_pitch_class_histogram",
        "pitch_class_transition_matrix", "note_length_hist",
        "note_length_transition_matrix",
    ]
    mg_rows = []
    for feat in mg_features:
        for m in _MODELS:
            mg_rows.append({
                "feature": feat, "model": m,
                "kl": "0.5", "oa": "0.5",
                "n_real_pieces": "337", "n_gen_pieces": "500",
            })
    _write_csv(
        metrics_dir / "mgeval.csv",
        header=["feature", "model", "kl", "oa", "n_real_pieces", "n_gen_pieces"],
        rows=mg_rows,
    )

    _write_csv(
        metrics_dir / "bar_rhythm_jsd.csv",
        header=["model", "jsd", "n_real_bars", "n_gen_bars",
                "n_unique_real", "n_unique_gen", "n_unique_union"],
        rows=[
            {"model": m, "jsd": "0.65", "n_real_bars": "2696",
             "n_gen_bars": "8000", "n_unique_real": "2022",
             "n_unique_gen": "1500", "n_unique_union": "3500"}
            for m in _MODELS
        ],
    )

    _write_csv(
        metrics_dir / "plagiarism.csv",
        header=[
            "model",
            "ngram_overlap_n3", "ngram_overlap_n4", "ngram_overlap_n5",
            "lcs_max_mean", "lcs_max_std", "lcs_max_median",
            "lcs_max_p25", "lcs_max_p75",
            "lcs_max_min", "lcs_max_max",
            "n_gen_chunks", "n_gen_chunks_lcs",
            "n_gen_ngrams_n3", "n_gen_ngrams_n4", "n_gen_ngrams_n5",
            "n_train_pieces",
        ],
        rows=[{
            "model": m,
            "ngram_overlap_n3": "0.9", "ngram_overlap_n4": "0.8",
            "ngram_overlap_n5": "0.5",
            "lcs_max_mean": "6.0", "lcs_max_std": "1.0",
            "lcs_max_median": "6.0",
            "lcs_max_p25": "5.0", "lcs_max_p75": "7.0",
            "lcs_max_min": "1", "lcs_max_max": "13",
            "n_gen_chunks": "500", "n_gen_chunks_lcs": "490",
            "n_gen_ngrams_n3": "10000", "n_gen_ngrams_n4": "9000",
            "n_gen_ngrams_n5": "8000",
            "n_train_pieces": "340",
        } for m in _MODELS],
    )


def test_build_total_72_rows(tmp_path):
    """24 метрики × 3 модели = 72 строки."""
    _setup_full_inputs(tmp_path / "_metrics")
    rows = FinalTableBuilder(tmp_path / "_metrics").build()
    assert len(rows) == 72


def test_build_order_follows_registry(tmp_path):
    """Метрики в порядке registry, модели в порядке MODEL_NAMES внутри каждой метрики."""
    _setup_full_inputs(tmp_path / "_metrics")
    rows = FinalTableBuilder(tmp_path / "_metrics").build()

    for i, mdef in enumerate(FINAL_METRICS):
        for j, model in enumerate(_MODELS):
            row = rows[i * 3 + j]
            assert row.metric == mdef.name, (
                f"row {i*3+j}: expected metric {mdef.name}, got {row.metric}"
            )
            assert row.model == model


def test_build_missing_metric_in_source_raises(tmp_path):
    """Если в registry есть метрика, а в её источнике её нет — fail-fast."""
    _setup_full_inputs(tmp_path / "_metrics")
    agg_path = tmp_path / "_metrics" / "aggregates.csv"
    with agg_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["metric"] != "note_density"]
    _write_csv(
        agg_path,
        header=["model", "metric", "n_themes", "mean", "std", "median",
                "p25", "p75", "min", "max"],
        rows=rows,
    )
    with pytest.raises(ValueError, match="note_density"):
        FinalTableBuilder(tmp_path / "_metrics").build()
