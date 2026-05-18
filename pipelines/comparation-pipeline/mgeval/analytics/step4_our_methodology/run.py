"""Step 4 — Наша целевая методология.

Тонкий wrapper над `compute_metrics.py` + bootstrap для Группы B.

Шаги:
  1. Standard pipeline → outputs/<slug>/_metrics/mgeval.csv (single-shot
     KL/OA по 9 признакам × 3 модели). Запускается отдельно:
        python pipelines/comparation-pipeline/compute_metrics.py \\
            --slug full-cleared-2samples
  2. Этот скрипт: Group B bootstrap (20 trials × N=15 random real-кусков
     vs полные gen-корпуса). Real/gen загружаются через ТЕ ЖЕ функции
     `load_real_corpus_chunks` + `iter_generated_corpus_chunks` +
     `Manifest.load`, что и `_compute_and_write_mgeval` — даёт ту же
     intersection-семантику (одно множество тем для всех моделей).
  3. Сшить mgeval.csv (Group A) + bootstrap (Group B) → result.csv.

Запуск (после `compute_metrics.py --slug full-cleared-2samples`):
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step4_our_methodology/run.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pretty_midi

COMP_ROOT = Path(__file__).resolve().parents[3]
if str(COMP_ROOT) not in sys.path:
    sys.path.insert(0, str(COMP_ROOT))

from manifest import Manifest  # noqa: E402
from mgeval.corpus_loader import (  # noqa: E402
    iter_generated_corpus_chunks,
    load_real_corpus_chunks,
)
from mgeval.features import FEATURES  # noqa: E402
from mgeval.pipeline import compute_mgeval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
SPLIT_JSON = REPO_ROOT / "pipelines" / "training-pipeline" / "wjazzd_split.json"
XML_DIR = REPO_ROOT / "models" / "MINGUS" / "A_preprocessData" / "data" / "xml"
SLUG = "full-cleared-2samples"
SLUG_DIR = REPO_ROOT / "pipelines" / "comparation-pipeline" / "outputs" / SLUG
MGEVAL_CSV = SLUG_DIR / "_metrics" / "mgeval.csv"

MODELS = ["cmt", "mingus", "bebopnet"]
GROUP_A = {
    "total_used_note",
    "total_pitch_class_histogram",
    "pitch_class_transition_matrix",
    "avg_ioi",
    "note_length_transition_matrix",
}
GROUP_B = {
    "total_used_pitch",
    "pitch_range",
    "avg_pitch_interval",
    "note_length_hist",
}
BOOTSTRAP_SEEDS = [1000 + i for i in range(20)]
N_SAMPLE = 15
CHUNK_BARS = 8


def _load_group_a() -> dict[tuple[str, str], tuple[float, float]]:
    """Считать (kl, oa) для Group A × MODELS из стандартного mgeval.csv."""
    if not MGEVAL_CSV.is_file():
        raise SystemExit(
            f"{MGEVAL_CSV} not found — run first:\n"
            f"  pipelines/comparation-pipeline/.venv/bin/python "
            f"pipelines/comparation-pipeline/compute_metrics.py --slug {SLUG}"
        )
    out: dict[tuple[str, str], tuple[float, float]] = {}
    with MGEVAL_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["feature"] in GROUP_A:
                out[(r["feature"], r["model"])] = (float(r["kl"]), float(r["oa"]))
    return out


def _load_corpora() -> tuple[
    list[pretty_midi.PrettyMIDI],
    dict[str, list[pretty_midi.PrettyMIDI]],
    int,
]:
    """Один-в-один как `_compute_and_write_mgeval` в compute_metrics.py."""
    real_chunks = load_real_corpus_chunks(
        split_json_path=SPLIT_JSON,
        xml_dir=XML_DIR,
        chunk_bars=CHUNK_BARS,
    )

    manifest = Manifest.load(SLUG_DIR / "manifest.json")
    active_themes = [
        name for name, t in manifest.themes.items()
        if t.status == "ok" and not t.removed_from_corpus
    ]
    gen_corpora = {
        model: iter_generated_corpus_chunks(
            SLUG_DIR, model, manifest.samples_per_theme, active_themes,
        )
        for model in MODELS
    }
    return real_chunks, gen_corpora, len(active_themes)


def _bootstrap_group_b(
    real_chunks: list[pretty_midi.PrettyMIDI],
    gen_corpora: dict[str, list[pretty_midi.PrettyMIDI]],
) -> dict[tuple[str, str], dict[str, float]]:
    """20 trials × MGEval(15 random real, gen_corpora) → mean/std/min/max
    для каждой пары (feature, model) ∈ GROUP_B × MODELS.
    """
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for trial_idx, seed in enumerate(BOOTSTRAP_SEEDS, 1):
        rng = random.Random(seed)
        sampled = rng.sample(real_chunks, N_SAMPLE)
        rows = compute_mgeval(sampled, gen_corpora)
        for r in rows:
            if r["feature"] in GROUP_B:
                key = (r["feature"], r["model"])
                acc.setdefault(key, {"kl": [], "oa": []})["kl"].append(r["kl"])
                acc[key]["oa"].append(r["oa"])
        print(
            f"  [{trial_idx:2d}/{len(BOOTSTRAP_SEEDS)}] seed={seed}",
            file=sys.stderr,
        )

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, lists in acc.items():
        kls = np.asarray(lists["kl"], dtype=float)
        oas = np.asarray(lists["oa"], dtype=float)
        out[key] = {
            "kl_mean": float(kls.mean()), "kl_std": float(kls.std()),
            "kl_min": float(kls.min()), "kl_max": float(kls.max()),
            "oa_mean": float(oas.mean()), "oa_std": float(oas.std()),
            "oa_min": float(oas.min()), "oa_max": float(oas.max()),
            "n_trials": len(kls),
        }
    return out


def main() -> None:
    print("=== step 4 — Наша целевая методология ===", file=sys.stderr)
    print(f"slug_dir: {SLUG_DIR}", file=sys.stderr)
    print(f"mgeval.csv (Group A): {MGEVAL_CSV}", file=sys.stderr)
    print(f"bootstrap_seeds: {BOOTSTRAP_SEEDS}", file=sys.stderr)
    print(f"Group A: {sorted(GROUP_A)}", file=sys.stderr)
    print(f"Group B: {sorted(GROUP_B)}", file=sys.stderr)

    # 1) Group A — из стандартного mgeval.csv.
    print("\nloading Group A from mgeval.csv ...", file=sys.stderr)
    group_a = _load_group_a()
    print(f"  Group A: {len(group_a)} (feature, model) pairs loaded", file=sys.stderr)

    # 2) Group B — bootstrap. Корпуса грузим через те же функции, что
    # compute_metrics.py:_compute_and_write_mgeval.
    print("\nloading corpora (same loaders as _compute_and_write_mgeval) ...",
          file=sys.stderr)
    real_chunks, gen_corpora, n_themes = _load_corpora()
    print(f"  real_chunks: {len(real_chunks)}", file=sys.stderr)
    for mdl in MODELS:
        print(f"  gen[{mdl}]: {len(gen_corpora[mdl])} chunks", file=sys.stderr)
    print(
        f"\nSANITY CHECK: N_themes = {n_themes} (одинаковое для всех 3 моделей "
        f"по Manifest intersection: status=ok AND not removed_from_corpus)",
        file=sys.stderr,
    )

    print(
        f"\nrunning bootstrap (Group B, {len(BOOTSTRAP_SEEDS)} trials, "
        f"N={N_SAMPLE} real chunks per trial) ...",
        file=sys.stderr,
    )
    group_b = _bootstrap_group_b(real_chunks, gen_corpora)

    # 3) Сшивка → result.csv (27 строк).
    print("\nwriting result.csv ...", file=sys.stderr)
    out_path = Path(__file__).resolve().parent / "result.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature", "group", "model",
            "kl_mean", "kl_std", "kl_min", "kl_max",
            "oa_mean", "oa_std", "oa_min", "oa_max",
            "n_trials",
        ])
        for feat in FEATURES.keys():
            if feat in GROUP_A:
                for mdl in MODELS:
                    kl, oa = group_a[(feat, mdl)]
                    w.writerow([
                        feat, "A", mdl,
                        f"{kl:.6f}", f"{0.0:.6f}", f"{kl:.6f}", f"{kl:.6f}",
                        f"{oa:.6f}", f"{0.0:.6f}", f"{oa:.6f}", f"{oa:.6f}",
                        1,
                    ])
            elif feat in GROUP_B:
                for mdl in MODELS:
                    agg = group_b[(feat, mdl)]
                    w.writerow([
                        feat, "B", mdl,
                        f"{agg['kl_mean']:.6f}", f"{agg['kl_std']:.6f}",
                        f"{agg['kl_min']:.6f}", f"{agg['kl_max']:.6f}",
                        f"{agg['oa_mean']:.6f}", f"{agg['oa_std']:.6f}",
                        f"{agg['oa_min']:.6f}", f"{agg['oa_max']:.6f}",
                        agg["n_trials"],
                    ])
            else:
                raise RuntimeError(f"feature {feat!r} вне Group A/B")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
