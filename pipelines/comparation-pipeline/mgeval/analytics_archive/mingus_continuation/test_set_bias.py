"""Test-set bias check для mingus_continuation.

Гипотеза: устойчивое расхождение по pitch-признакам (PC, PR, PI — 2-15× хуже
paper-MINGUS) объясняется тем, что наша случайная подвыборка 15 WjazzD-соло
≠ их случайная. Проверяется прогоном на N разных random-15 выборках:
- если KL/OA сильно варьируется между trial'ами → гипотеза подтверждается
  (числа были бы другими на другой выборке).
- если стабильны → bias не в выборке, а в моделях.

Параметры: N=5 trials, default 4-bar seed → 12-bar gen (как matched-режим
в `mgeval_continuation.csv`). random_seeds = [42, 123, 7, 999, 2024].

Артефакты в trial_<seed>/ subdirs (в .gitignore). Inputs регенерятся при
повторном запуске. CSV-результат рядом со скриптом.

Output:
    test_set_bias_per_trial.csv  — длинный формат: trial, feature,
                                   kl_existing, oa_existing, kl_ours, oa_ours
    test_set_bias_aggregate.csv  — на каждую (feature, checkpoint):
                                   mean, std, min, max по 5 trials
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT_DIR = THIS.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np

# Импортируем из run.py — переиспользуем prepare_inputs, run_mingus,
# load_pms, конфиги путей/чекпоинтов.
from run import (
    CKPT_EXISTING, CKPT_OURS, MINGUS_DATA, MINGUS_FORK,
    load_pms, prepare_inputs, run_mingus,
)

# pylint: disable=wrong-import-position
COMP_ROOT = THIS.parents[3]
sys.path.insert(0, str(COMP_ROOT))

from mgeval.pipeline import compute_mgeval  # noqa: E402
from models.mingus import GeneratorMingus  # noqa: E402

N_SAMPLES = 15
INPUT_BARS = 4
OUTPUT_BARS = 12
RANDOM_SEEDS = [42, 123, 7, 999, 2024]

PER_TRIAL_CSV = SCRIPT_DIR / "test_set_bias_per_trial.csv"
AGGREGATE_CSV = SCRIPT_DIR / "test_set_bias_aggregate.csv"


def run_one_trial(random_seed: int) -> list[dict]:
    """Один trial: random выборка 15 WjazzD → MINGUS на двух чекпоинтах
    → MGEval. Возвращает list[dict] с keys feature/model/kl/oa.
    """
    trial_dir = SCRIPT_DIR / f"trial_{random_seed}"
    inputs_dir = trial_dir / "inputs"
    real_dir = trial_dir / "real"
    gen_existing_dir = trial_dir / "gen_existing"
    gen_ours_dir = trial_dir / "gen_ours"

    print(f"\n=== trial random_seed={random_seed} ===", flush=True)
    print(f"  prepare inputs in {trial_dir}", flush=True)
    inputs = prepare_inputs(
        N_SAMPLES, INPUT_BARS, inputs_dir, real_dir,
        random_seed=random_seed,
    )
    print(f"  prepared {len(inputs)} inputs", flush=True)

    print("  MINGUS existing_checkpoint", flush=True)
    gen_existing = GeneratorMingus(
        fork_root=MINGUS_FORK,
        data_path=MINGUS_DATA,
        checkpoint_dir=CKPT_EXISTING,
        epochs=100,
        cond_pitch="I-C-NC-B-BE-O",
        cond_duration="I-C-NC-B-BE-O",
        device="cpu",
    )
    try:
        gen_existing_paths = run_mingus(
            gen_existing, inputs, gen_existing_dir, INPUT_BARS, OUTPUT_BARS,
        )
    finally:
        gen_existing.close()

    print("  MINGUS our_checkpoint", flush=True)
    gen_ours = GeneratorMingus(
        fork_root=MINGUS_FORK,
        data_path=MINGUS_DATA,
        checkpoint_dir=CKPT_OURS,
        epochs=10,
        cond_pitch="D-C-B-BE-O",
        cond_duration="B-BE-O",
        device="cpu",
    )
    try:
        gen_ours_paths = run_mingus(
            gen_ours, inputs, gen_ours_dir, INPUT_BARS, OUTPUT_BARS,
        )
    finally:
        gen_ours.close()

    real_pms = load_pms([t[2] for t in inputs])
    gen_existing_pms = load_pms(gen_existing_paths)
    gen_ours_pms = load_pms(gen_ours_paths)
    print(f"  loaded real={len(real_pms)} existing={len(gen_existing_pms)} "
          f"ours={len(gen_ours_pms)}", flush=True)

    rows = compute_mgeval(
        real_pms,
        {
            "existing_checkpoint": gen_existing_pms,
            "our_checkpoint": gen_ours_pms,
        },
    )
    return rows


def main() -> int:
    print(f"script dir: {SCRIPT_DIR}", flush=True)
    print(f"running {len(RANDOM_SEEDS)} trials with seeds {RANDOM_SEEDS}",
          flush=True)
    print(f"params: n_samples={N_SAMPLES} input_bars={INPUT_BARS} "
          f"output_bars={OUTPUT_BARS}", flush=True)

    all_rows: list[dict] = []  # каждая запись: trial, feature, model, kl, oa
    for seed in RANDOM_SEEDS:
        trial_rows = run_one_trial(seed)
        for r in trial_rows:
            all_rows.append({"trial_seed": seed, **r})

    # 1) per_trial CSV (long format)
    per_trial_data: dict[tuple[int, str], dict] = {}
    for r in all_rows:
        key = (r["trial_seed"], r["feature"])
        d = per_trial_data.setdefault(
            key, {"trial_seed": r["trial_seed"], "feature": r["feature"]}
        )
        if r["model"] == "existing_checkpoint":
            d["kl_existing"] = f"{r['kl']:.6f}"
            d["oa_existing"] = f"{r['oa']:.6f}"
        else:
            d["kl_ours"] = f"{r['kl']:.6f}"
            d["oa_ours"] = f"{r['oa']:.6f}"

    per_trial_fields = [
        "trial_seed", "feature",
        "kl_existing", "oa_existing",
        "kl_ours", "oa_ours",
    ]
    with PER_TRIAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_trial_fields)
        w.writeheader()
        # Стабильный порядок: trial × feature
        for seed in RANDOM_SEEDS:
            for r in all_rows:
                if r["trial_seed"] != seed or r["model"] != "existing_checkpoint":
                    continue
                w.writerow(per_trial_data[(seed, r["feature"])])
    print(f"\nwrote {PER_TRIAL_CSV}", flush=True)

    # 2) aggregate CSV: mean/std/min/max per (feature × checkpoint) по trials.
    features_order: list[str] = []
    for r in all_rows:
        if r["feature"] not in features_order:
            features_order.append(r["feature"])

    agg_fields = [
        "feature",
        "kl_existing_mean", "kl_existing_std",
        "kl_existing_min", "kl_existing_max",
        "oa_existing_mean", "oa_existing_std",
        "oa_existing_min", "oa_existing_max",
        "kl_ours_mean", "kl_ours_std",
        "kl_ours_min", "kl_ours_max",
        "oa_ours_mean", "oa_ours_std",
        "oa_ours_min", "oa_ours_max",
    ]
    with AGGREGATE_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for feat in features_order:
            row: dict = {"feature": feat}
            for model_key, suffix in [
                ("existing_checkpoint", "existing"),
                ("our_checkpoint", "ours"),
            ]:
                kls = np.array([
                    r["kl"] for r in all_rows
                    if r["feature"] == feat and r["model"] == model_key
                ])
                oas = np.array([
                    r["oa"] for r in all_rows
                    if r["feature"] == feat and r["model"] == model_key
                ])
                row[f"kl_{suffix}_mean"] = f"{float(kls.mean()):.6f}"
                row[f"kl_{suffix}_std"] = f"{float(kls.std()):.6f}"
                row[f"kl_{suffix}_min"] = f"{float(kls.min()):.6f}"
                row[f"kl_{suffix}_max"] = f"{float(kls.max()):.6f}"
                row[f"oa_{suffix}_mean"] = f"{float(oas.mean()):.6f}"
                row[f"oa_{suffix}_std"] = f"{float(oas.std()):.6f}"
                row[f"oa_{suffix}_min"] = f"{float(oas.min()):.6f}"
                row[f"oa_{suffix}_max"] = f"{float(oas.max()):.6f}"
            w.writerow(row)
    print(f"wrote {AGGREGATE_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
