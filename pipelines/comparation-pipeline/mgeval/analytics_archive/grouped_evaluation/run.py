"""Grouped MGEval-пересчёт на full-cleared-2samples.

Делит 9 признаков на две группы по чувствительности к test-set bias
(см. paper/comparation/mgeval_diagnostics.md, Эксперимент 4):

- Группа A (стабильные): один прогон, real-корпус целиком (337 chunks).
- Группа B (нестабильные): K=20 random subsamples real-корпуса по N=15,
  для каждого — KL/OA, итог — mean/std/min/max.

Output:
    pipelines/comparation-pipeline/mgeval/baselines/grouped_evaluation/mgeval_grouped.csv
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
import pretty_midi  # noqa: F401 — нужен для typing хелперов ниже

from manifest import Manifest
from mgeval.corpus_loader import (
    iter_generated_corpus_chunks, load_real_corpus_chunks,
)
from mgeval.distances import c_dist, kl_dist, overlap_area
from mgeval.features import FEATURES

# --- Конфигурация ---
SLUG_DIR = COMP_ROOT / "outputs/full-cleared-2samples"
SPLIT_JSON = REPO_ROOT / "pipelines/training-pipeline/wjazzd_split.json"
REAL_XML_DIR = REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml"
CHUNK_BARS = 8
MODELS = ("cmt", "mingus", "bebopnet")

GROUP_A = (
    "total_used_note",
    "avg_ioi",
    "note_length_hist",
    "note_length_transition_matrix",
    "total_pitch_class_histogram",
    "pitch_class_transition_matrix",
)
GROUP_B = (
    "total_used_pitch",
    "pitch_range",
    "avg_pitch_interval",
)

BOOTSTRAP_N_TRIALS = 20
BOOTSTRAP_SAMPLE_SIZE = 15

OUT_CSV = SCRIPT_DIR / "mgeval_grouped.csv"


# --- Вспомогательные ---


def _extract_features(corpus, feature_fn) -> np.ndarray:
    vectors = []
    for pm in corpus:
        v = feature_fn(pm)
        if v is None:
            continue
        v = np.asarray(v).flatten().astype(float)
        vectors.append(v)
    if not vectors:
        raise ValueError("all extractions returned None")
    return np.stack(vectors, axis=0)


def _intra_distances(X: np.ndarray) -> np.ndarray:
    n = len(X)
    if n < 2:
        raise ValueError(f"intra requires >=2 samples, got {n}")
    out = np.zeros((n, n - 1))
    for i in range(n):
        mask = np.arange(n) != i
        out[i] = c_dist(X[i], X[mask])
    return out.flatten()


def _inter_distances(X_real: np.ndarray, X_gen: np.ndarray) -> np.ndarray:
    out = np.zeros((len(X_real), len(X_gen)))
    for i in range(len(X_real)):
        out[i] = c_dist(X_real[i], X_gen)
    return out.flatten()


def _kl_oa_for_feature(real_pms, gen_pms, feature_fn) -> tuple[float, float]:
    X_real = _extract_features(real_pms, feature_fn)
    X_gen = _extract_features(gen_pms, feature_fn)
    intra = _intra_distances(X_real)
    inter = _inter_distances(X_real, X_gen)
    return kl_dist(intra, inter), overlap_area(intra, inter)


# --- Группа A: один прогон ---


def compute_group_a(real_pms, gen_corpora) -> list[dict]:
    rows: list[dict] = []
    for feat in GROUP_A:
        fn = FEATURES[feat]
        for model in MODELS:
            kl, oa = _kl_oa_for_feature(real_pms, gen_corpora[model], fn)
            rows.append({
                "feature": feat, "group": "A", "model": model,
                "kl_mean": kl, "kl_std": 0.0,
                "kl_min": kl, "kl_max": kl,
                "oa_mean": oa, "oa_std": 0.0,
                "oa_min": oa, "oa_max": oa,
                "n_trials": 1,
            })
            print(f"  A: {feat:35s} {model:9s} kl={kl:7.4f} oa={oa:6.4f}",
                  flush=True)
    return rows


# --- Группа B: bootstrap ---


def compute_group_b(real_pms, gen_corpora) -> list[dict]:
    rng = np.random.default_rng(42)
    # Накопители: feature × model → list of kl, list of oa.
    kl_acc: dict[tuple[str, str], list[float]] = {}
    oa_acc: dict[tuple[str, str], list[float]] = {}
    for feat in GROUP_B:
        for model in MODELS:
            kl_acc[(feat, model)] = []
            oa_acc[(feat, model)] = []

    for trial in range(BOOTSTRAP_N_TRIALS):
        idx = rng.choice(len(real_pms), size=BOOTSTRAP_SAMPLE_SIZE, replace=False)
        real_sample = [real_pms[i] for i in idx]
        print(f"\n  trial {trial+1}/{BOOTSTRAP_N_TRIALS} "
              f"(sample of {len(real_sample)} real)", flush=True)
        for feat in GROUP_B:
            fn = FEATURES[feat]
            for model in MODELS:
                kl, oa = _kl_oa_for_feature(real_sample, gen_corpora[model], fn)
                kl_acc[(feat, model)].append(kl)
                oa_acc[(feat, model)].append(oa)
                print(f"    B: {feat:25s} {model:9s} kl={kl:7.4f} oa={oa:6.4f}",
                      flush=True)

    rows: list[dict] = []
    for feat in GROUP_B:
        for model in MODELS:
            kls = np.array(kl_acc[(feat, model)])
            oas = np.array(oa_acc[(feat, model)])
            rows.append({
                "feature": feat, "group": "B", "model": model,
                "kl_mean": float(kls.mean()), "kl_std": float(kls.std()),
                "kl_min": float(kls.min()), "kl_max": float(kls.max()),
                "oa_mean": float(oas.mean()), "oa_std": float(oas.std()),
                "oa_min": float(oas.min()), "oa_max": float(oas.max()),
                "n_trials": BOOTSTRAP_N_TRIALS,
            })
    return rows


# --- main ---


def main() -> int:
    print(f"slug: {SLUG_DIR}", flush=True)
    print(f"output: {OUT_CSV}\n", flush=True)

    print("=== loading real corpus (40 wjazzd-test → 8-bar chunks) ===", flush=True)
    real_pms = load_real_corpus_chunks(
        split_json_path=SPLIT_JSON,
        xml_dir=REAL_XML_DIR,
        chunk_bars=CHUNK_BARS,
    )
    print(f"real corpus: {len(real_pms)} chunks", flush=True)

    print("\n=== loading generated corpora ===", flush=True)
    manifest = Manifest.load(SLUG_DIR / "manifest.json")
    active_themes = [
        name for name, t in manifest.themes.items()
        if t.status == "ok" and not t.removed_from_corpus
    ]
    gen_corpora = {}
    for model in MODELS:
        chunks = iter_generated_corpus_chunks(
            SLUG_DIR, model, manifest.samples_per_theme, active_themes,
        )
        gen_corpora[model] = chunks
        print(f"  {model}: {len(chunks)} chunks", flush=True)

    print("\n=== Group A (стабильные фичи, 1 прогон) ===", flush=True)
    rows_a = compute_group_a(real_pms, gen_corpora)

    print("\n=== Group B (нестабильные фичи, "
          f"{BOOTSTRAP_N_TRIALS} bootstrap trials × N={BOOTSTRAP_SAMPLE_SIZE}) ===",
          flush=True)
    rows_b = compute_group_b(real_pms, gen_corpora)

    fieldnames = [
        "feature", "group", "model",
        "kl_mean", "kl_std", "kl_min", "kl_max",
        "oa_mean", "oa_std", "oa_min", "oa_max",
        "n_trials",
    ]
    rows = rows_a + rows_b
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                **r,
                "kl_mean": f"{r['kl_mean']:.6f}",
                "kl_std": f"{r['kl_std']:.6f}",
                "kl_min": f"{r['kl_min']:.6f}",
                "kl_max": f"{r['kl_max']:.6f}",
                "oa_mean": f"{r['oa_mean']:.6f}",
                "oa_std": f"{r['oa_std']:.6f}",
                "oa_min": f"{r['oa_min']:.6f}",
                "oa_max": f"{r['oa_max']:.6f}",
            })
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows: "
          f"{len(rows_a)} group A + {len(rows_b)} group B)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
