"""Step 3 — Test-set bias study.

5 trials × {existing, ours} MINGUS-чекпоинт на разных random_sample(15
test-соло) → MGEval whole-solos → per_trial.csv + aggregate.csv.

Логика efficient-run: для каждой конфигурации (existing / ours) MINGUS
запускается ровно один раз на union всех нужных файлов. Persistent
subprocess не перезагружает чекпоинт между запросами; внутри runner'а
`torch.manual_seed(seed)` ресетит RNG в начале каждой генерации, поэтому
результат для (solo, seed) детерминирован независимо от порядка.

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step3_test_set_bias/run.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Подтащить _step2_common.py — там фабрики генераторов, run_mingus_on_solos,
# load_real_corpus, select_solo_names.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step2_paper_reproduction"))

from _step2_common import (  # noqa: E402
    GEN_SEED,
    load_real_corpus,
    make_existing_generator,
    make_ours_generator,
    run_mingus_on_solos,
    select_solo_names,
)
from mgeval.features import FEATURES  # noqa: E402
from mgeval.pipeline import compute_mgeval  # noqa: E402

TRIAL_SEEDS = [42, 123, 7, 999, 2024]
N_SOLOS = 15
INPUT_BARS = 4  # длинный seed (как в 2b)
OUTPUT_BARS = 12


def main() -> None:
    print("=== step 3 — Test-set bias study ===", file=sys.stderr)
    print(
        f"trial_seeds={TRIAL_SEEDS}, n_solos={N_SOLOS}, "
        f"input_bars={INPUT_BARS}, output_bars={OUTPUT_BARS}, gen_seed={GEN_SEED}",
        file=sys.stderr,
    )

    # 1) Для каждого trial — отобрать 15 имён.
    selected_per_trial: dict[int, list[str]] = {
        seed: select_solo_names(seed=seed, n=N_SOLOS) for seed in TRIAL_SEEDS
    }
    print("\nselected per trial:", file=sys.stderr)
    for seed in TRIAL_SEEDS:
        print(f"  trial seed={seed}:", file=sys.stderr)
        for name in selected_per_trial[seed]:
            print(f"    {name}", file=sys.stderr)

    # 2) Union — прогнать каждую конфигурацию один раз на всех уникальных файлах.
    union_names = sorted({n for names in selected_per_trial.values() for n in names})
    print(f"\nunion across trials: {len(union_names)} unique solos", file=sys.stderr)

    print("\nrunning MINGUS (existing — paper checkpoint) on union ...", file=sys.stderr)
    gen_existing = make_existing_generator()
    try:
        ok_existing, failed_existing = run_mingus_on_solos(
            gen_existing, union_names, INPUT_BARS, OUTPUT_BARS, GEN_SEED,
            log_prefix="existing ",
        )
    finally:
        gen_existing.close()

    print("\nrunning MINGUS (ours — paper-optimal) on union ...", file=sys.stderr)
    gen_ours = make_ours_generator()
    try:
        ok_ours, failed_ours = run_mingus_on_solos(
            gen_ours, union_names, INPUT_BARS, OUTPUT_BARS, GEN_SEED,
            log_prefix="ours     ",
        )
    finally:
        gen_ours.close()

    # 3) MGEval per trial, со skip-and-keep-pair.
    per_trial_records: list[dict] = []
    agg: dict[str, dict[str, list[float]]] = {
        feat: {"kl_e": [], "oa_e": [], "kl_o": [], "oa_o": []}
        for feat in FEATURES.keys()
    }

    for trial_seed in TRIAL_SEEDS:
        selected = selected_per_trial[trial_seed]
        surviving = [n for n in selected if n in ok_existing and n in ok_ours]
        dropped = [n for n in selected if n not in surviving]
        if dropped:
            print(
                f"\ntrial seed={trial_seed}: dropped {len(dropped)} (failed in some config):",
                file=sys.stderr,
            )
            for n in dropped:
                tag_e = "FAIL" if n in failed_existing else "ok"
                tag_o = "FAIL" if n in failed_ours else "ok"
                print(f"  {n} [existing={tag_e}, ours={tag_o}]", file=sys.stderr)

        real_pms = load_real_corpus(surviving)
        gen_e = [ok_existing[n] for n in surviving]
        gen_o = [ok_ours[n] for n in surviving]
        print(
            f"\ntrial seed={trial_seed}: real={len(real_pms)} "
            f"existing={len(gen_e)} ours={len(gen_o)} — running MGEval ...",
            file=sys.stderr,
        )

        rows = compute_mgeval(real_pms, {"existing": gen_e, "ours": gen_o})
        by_fm = {(r["feature"], r["model"]): (r["kl"], r["oa"]) for r in rows}

        for feat in FEATURES.keys():
            kl_e, oa_e = by_fm[(feat, "existing")]
            kl_o, oa_o = by_fm[(feat, "ours")]
            per_trial_records.append({
                "trial_seed": trial_seed,
                "feature": feat,
                "kl_existing": kl_e,
                "oa_existing": oa_e,
                "kl_ours": kl_o,
                "oa_ours": oa_o,
                "n_solos": len(surviving),
            })
            agg[feat]["kl_e"].append(kl_e)
            agg[feat]["oa_e"].append(oa_e)
            agg[feat]["kl_o"].append(kl_o)
            agg[feat]["oa_o"].append(oa_o)

    # 4) Write per_trial.csv: 45 строк = 5 trial × 9 feat.
    out_dir = Path(__file__).resolve().parent
    per_trial_path = out_dir / "per_trial.csv"
    with per_trial_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_seed", "feature",
            "kl_existing", "oa_existing", "kl_ours", "oa_ours",
            "n_solos",
        ])
        for r in per_trial_records:
            w.writerow([
                r["trial_seed"], r["feature"],
                f"{r['kl_existing']:.6f}", f"{r['oa_existing']:.6f}",
                f"{r['kl_ours']:.6f}", f"{r['oa_ours']:.6f}",
                r["n_solos"],
            ])
    print(f"\nwrote {per_trial_path}", file=sys.stderr)

    # 5) Write aggregate.csv: 9 строк (по feature) с агрегатами по 5 trial.
    agg_path = out_dir / "aggregate.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature",
            "kl_existing_mean", "kl_existing_std", "kl_existing_min", "kl_existing_max",
            "kl_ours_mean", "kl_ours_std", "kl_ours_min", "kl_ours_max",
            "oa_existing_mean", "oa_ours_mean",
        ])
        for feat in FEATURES.keys():
            kl_e = np.asarray(agg[feat]["kl_e"], dtype=float)
            kl_o = np.asarray(agg[feat]["kl_o"], dtype=float)
            oa_e = np.asarray(agg[feat]["oa_e"], dtype=float)
            oa_o = np.asarray(agg[feat]["oa_o"], dtype=float)
            w.writerow([
                feat,
                f"{kl_e.mean():.6f}", f"{kl_e.std():.6f}", f"{kl_e.min():.6f}", f"{kl_e.max():.6f}",
                f"{kl_o.mean():.6f}", f"{kl_o.std():.6f}", f"{kl_o.min():.6f}", f"{kl_o.max():.6f}",
                f"{oa_e.mean():.6f}", f"{oa_o.mean():.6f}",
            ])
    print(f"wrote {agg_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
