"""Step 1b — Real-vs-Real baseline на 8-bar чанках.

40 test-соло → нарезать на непересекающиеся 8-bar окна (хвосты <8 тактов
отбрасываются). 5 trials × shuffle(chunks) → split пополам → MGEval(A, B)
→ агрегировать KL/OA по 9 признакам → result.csv.

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step1_baseline/1b_chunked/run.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _step1_common import (  # noqa: E402
    SEEDS,
    aggregate_rows,
    load_chunks_8bar,
    write_result_csv,
)
from mgeval.pipeline import compute_mgeval  # noqa: E402


def main() -> None:
    print("=== step 1b — Chunked baseline (8-bar) ===", file=sys.stderr)
    chunks, counts = load_chunks_8bar()
    print(f"loaded {len(chunks)} chunks из {len(counts)} соло:", file=sys.stderr)
    for name, n in counts.items():
        print(f"  {name}: {n} chunks", file=sys.stderr)

    print(f"\nseeds: {SEEDS}", file=sys.stderr)
    per_trial = []
    for seed in SEEDS:
        rng = random.Random(seed)
        order = list(chunks)
        rng.shuffle(order)
        half = len(order) // 2
        a_corpus = order[:half]
        b_corpus = order[half:]
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
