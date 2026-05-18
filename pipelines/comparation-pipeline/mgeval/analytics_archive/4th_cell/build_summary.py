"""Собирает сводную таблицу сравнения 4 клеток 2×2 (slicing × input)
для MINGUS-our из 4 источников.

Источники:
- chunked+WjazzD: mgeval_4th_cell.csv (этот эксперимент)
- full+WjazzD:    mingus_continuation/mgeval_continuation_longseed.csv
                  (Эксп. 3 long-seed continuation, кол-ка kl_ours/oa_ours)
- full+effendi:   _sources/mgeval_full_solos.csv (snapshot из /tmp прогона на
                  effendi-fakebook полных gen_full.mid; model=mingus)
- chunked+effendi: grouped_evaluation/mgeval_grouped.csv (group A/B для mingus)

Output: summary_2x2.csv (рядом со скриптом).
"""
from __future__ import annotations

import csv
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT_DIR = THIS.parent
REPO_ROOT = THIS.parents[5]
BASELINES = REPO_ROOT / "pipelines/comparation-pipeline/mgeval/baselines"

CELL_4TH = SCRIPT_DIR / "mgeval_4th_cell.csv"
FULL_WJAZZD = BASELINES / "mingus_continuation/mgeval_continuation_longseed.csv"
FULL_EFFENDI = SCRIPT_DIR / "_sources/mgeval_full_solos.csv"
CHUNKED_EFFENDI = BASELINES / "grouped_evaluation/mgeval_grouped.csv"

OUT_CSV = SCRIPT_DIR / "summary_2x2.csv"


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARN: {path} not found — оставлю эти ячейки пустыми",
              flush=True)
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows_4th = _load_csv(CELL_4TH)
    rows_full_w = _load_csv(FULL_WJAZZD)
    rows_full_e = _load_csv(FULL_EFFENDI)
    rows_chunked_e = _load_csv(CHUNKED_EFFENDI)

    summary: dict[str, dict] = {}

    # chunked+WjazzD
    for r in rows_4th:
        feat = r["feature"]
        summary.setdefault(feat, {"feature": feat})
        summary[feat]["chunked_wjazzd_kl"] = r["kl_chunked_wjazzd"]
        summary[feat]["chunked_wjazzd_oa"] = r["oa_chunked_wjazzd"]
        if r.get("kl_bootstrap_mean"):
            summary[feat]["chunked_wjazzd_kl_bootstrap_mean"] = r["kl_bootstrap_mean"]
            summary[feat]["chunked_wjazzd_kl_bootstrap_std"] = r["kl_bootstrap_std"]

    # full+WjazzD (берём колонки kl_ours/oa_ours для нашего чекпоинта)
    for r in rows_full_w:
        feat = r["feature"]
        summary.setdefault(feat, {"feature": feat})
        summary[feat]["full_wjazzd_kl"] = r["kl_ours"]
        summary[feat]["full_wjazzd_oa"] = r["oa_ours"]

    # full+effendi (model=mingus)
    for r in rows_full_e:
        if r.get("model") != "mingus":
            continue
        feat = r["feature"]
        summary.setdefault(feat, {"feature": feat})
        summary[feat]["full_effendi_kl"] = r["kl"]
        summary[feat]["full_effendi_oa"] = r["oa"]

    # chunked+effendi (model=mingus, group A = single, group B = bootstrap mean)
    for r in rows_chunked_e:
        if r.get("model") != "mingus":
            continue
        feat = r["feature"]
        summary.setdefault(feat, {"feature": feat})
        summary[feat]["chunked_effendi_kl"] = r["kl_mean"]
        summary[feat]["chunked_effendi_oa"] = r["oa_mean"]
        if int(r["n_trials"]) > 1:
            summary[feat]["chunked_effendi_kl_bootstrap_std"] = r["kl_std"]

    fieldnames = [
        "feature",
        "full_wjazzd_kl", "full_wjazzd_oa",
        "chunked_wjazzd_kl", "chunked_wjazzd_oa",
        "chunked_wjazzd_kl_bootstrap_mean", "chunked_wjazzd_kl_bootstrap_std",
        "full_effendi_kl", "full_effendi_oa",
        "chunked_effendi_kl", "chunked_effendi_oa",
        "chunked_effendi_kl_bootstrap_std",
    ]
    feature_order = (
        "total_used_pitch", "total_pitch_class_histogram",
        "pitch_class_transition_matrix", "pitch_range",
        "avg_pitch_interval", "total_used_note", "avg_ioi",
        "note_length_hist", "note_length_transition_matrix",
    )
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for feat in feature_order:
            row = {"feature": feat}
            row.update(summary.get(feat, {}))
            for fn in fieldnames:
                row.setdefault(fn, "")
            w.writerow(row)
    print(f"wrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    main()
