"""Step 1a — Real-vs-Real baseline на целых соло.

5 trials × shuffle(40 solos) → split 20/20 → MGEval(A, B) → агрегировать
KL/OA по 9 признакам → result.csv.

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step1_baseline/1a_full_solos/run.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Подтащить _common.py из родительской папки.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _step1_common import (  # noqa: E402
    SEEDS,
    aggregate_rows,
    load_full_solos,
    write_result_csv,
)
from mgeval.pipeline import compute_mgeval  # noqa: E402


def main() -> None:
    print("=== step 1a — Full-solos baseline ===", file=sys.stderr)
    solos = load_full_solos()
    print(f"loaded {len(solos)} solos:", file=sys.stderr)
    for name, _ in solos:
        print(f"  {name}", file=sys.stderr)

    print(f"\nseeds: {SEEDS}", file=sys.stderr)
    per_trial = []
    for seed in SEEDS:
        rng = random.Random(seed)
        order = list(solos)
        rng.shuffle(order)
        a_corpus = [pm for _, pm in order[:20]]
        b_corpus = [pm for _, pm in order[20:]]
        print(
            f"seed={seed}: |A|={len(a_corpus)} |B|={len(b_corpus)}",
            file=sys.stderr,
        )
        rows = compute_mgeval(a_corpus, {"B": b_corpus})
        per_trial.append(rows)

    agg = aggregate_rows(per_trial)
    out_path = Path(__file__).resolve().parent / "result.csv"
    write_result_csv(out_path, agg)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
