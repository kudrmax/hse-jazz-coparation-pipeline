"""Chunked Real-vs-Real baseline (Эксперимент 1b).

Sanity-check для MGEval-pipeline в chunked-режиме: real-корпус нарезаем
на 8-bar chunks, случайно делим пополам, прогоняем MGEval. Ожидаем KL≈0,
OA≈1 — иначе у chunked-pipeline проблема.

5 trials с разными random_seeds (теми же что в test_set_bias эксперименте).

Output:
    chunked_baseline_per_trial.csv  — 45 строк (9 фич × 5 trials)
    chunked_baseline_aggregate.csv  — 9 строк (aggregate)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT_DIR = THIS.parent
REPO_ROOT = THIS.parents[5]
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"
sys.path.insert(0, str(COMP_ROOT))

import numpy as np

from mgeval.corpus_loader import load_real_corpus_chunks
from mgeval.pipeline import compute_mgeval

SPLIT_JSON = REPO_ROOT / "pipelines/training-pipeline/wjazzd_split.json"
REAL_XML_DIR = REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml"
CHUNK_BARS = 8
RANDOM_SEEDS = [42, 123, 7, 999, 2024]

PER_TRIAL_CSV = SCRIPT_DIR / "chunked_baseline_per_trial.csv"
AGGREGATE_CSV = SCRIPT_DIR / "chunked_baseline_aggregate.csv"


def main() -> int:
    print(f"output: {PER_TRIAL_CSV} + {AGGREGATE_CSV}\n", flush=True)

    print("=== loading real corpus (40 wjazzd-test → 8-bar chunks) ===",
          flush=True)
    real_chunks = load_real_corpus_chunks(
        split_json_path=SPLIT_JSON,
        xml_dir=REAL_XML_DIR,
        chunk_bars=CHUNK_BARS,
    )
    n_total = len(real_chunks)
    half = n_total // 2
    print(f"real corpus: {n_total} chunks → A={half} B={half} per trial",
          flush=True)

    all_rows: list[dict] = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        A_idx = perm[:half]
        B_idx = perm[half:half * 2]
        # Sanity: A и B не пересекаются (математически гарантировано slices
        # одного permutation, assert для уверенности).
        assert len(set(A_idx.tolist()) & set(B_idx.tolist())) == 0, "overlap!"
        A = [real_chunks[i] for i in A_idx]
        B = [real_chunks[i] for i in B_idx]

        print(f"\n=== trial seed={seed}: A={len(A)} B={len(B)} (disjoint) ===",
              flush=True)
        rows = compute_mgeval(A, {"B": B})
        for r in rows:
            all_rows.append({
                "trial_seed": seed,
                "feature": r["feature"],
                "kl": r["kl"],
                "oa": r["oa"],
                "n_real_chunks": len(A),
                "n_gen_chunks": len(B),
            })
            print(f"  {r['feature']:35s} kl={r['kl']:7.4f}  oa={r['oa']:6.4f}",
                  flush=True)

    # per_trial CSV
    per_trial_fields = [
        "trial_seed", "feature", "kl", "oa", "n_real_chunks", "n_gen_chunks",
    ]
    with PER_TRIAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_trial_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                **r,
                "kl": f"{r['kl']:.6f}",
                "oa": f"{r['oa']:.6f}",
            })
    print(f"\nwrote {PER_TRIAL_CSV} ({len(all_rows)} rows)", flush=True)

    # aggregate CSV: per-feature mean/std/min/max по 5 trials
    features_order: list[str] = []
    for r in all_rows:
        if r["feature"] not in features_order:
            features_order.append(r["feature"])
    agg_fields = [
        "feature",
        "kl_mean", "kl_std", "kl_min", "kl_max",
        "oa_mean", "oa_std", "oa_min", "oa_max",
    ]
    with AGGREGATE_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for feat in features_order:
            kls = np.array([r["kl"] for r in all_rows if r["feature"] == feat])
            oas = np.array([r["oa"] for r in all_rows if r["feature"] == feat])
            w.writerow({
                "feature": feat,
                "kl_mean": f"{float(kls.mean()):.6f}",
                "kl_std": f"{float(kls.std()):.6f}",
                "kl_min": f"{float(kls.min()):.6f}",
                "kl_max": f"{float(kls.max()):.6f}",
                "oa_mean": f"{float(oas.mean()):.6f}",
                "oa_std": f"{float(oas.std()):.6f}",
                "oa_min": f"{float(oas.min()):.6f}",
                "oa_max": f"{float(oas.max()):.6f}",
            })
    print(f"wrote {AGGREGATE_CSV} ({len(features_order)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
