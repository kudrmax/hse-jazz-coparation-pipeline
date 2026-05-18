"""Step 2b — Paper-MINGUS reproduction (длинный seed).

15 тех же случайных WjazzD-test соло (random_seed=42, что и в 2a) → для
каждого взять первые 4 такта (input_bars=4) как seed → MINGUS прогон в
двух конфигурациях (existing / ours). MGEval(15 real vs 15 generated)
для каждой → result.csv.

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step2_paper_reproduction/2b_long_seed/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _step2_common import (  # noqa: E402
    GEN_SEED,
    OUTPUT_BARS,
    load_real_corpus,
    make_existing_generator,
    make_ours_generator,
    run_mingus_on_solos,
    select_solo_names,
    write_step2_csv,
)
from mgeval.pipeline import compute_mgeval  # noqa: E402

INPUT_BARS = 4  # длинный seed — первые 4 такта


def main() -> None:
    print("=== step 2b — Paper-MINGUS reproduction (long seed) ===", file=sys.stderr)
    print(f"input_bars={INPUT_BARS}, output_bars={OUTPUT_BARS}, seed={GEN_SEED}", file=sys.stderr)

    selected = select_solo_names()
    print(f"\nselected {len(selected)} solos (random_seed=42):", file=sys.stderr)
    for name in selected:
        print(f"  {name}", file=sys.stderr)

    print("\nrunning MINGUS (existing — paper checkpoint, cond=I-C-NC-B-BE-O, epochs=100) ...",
          file=sys.stderr)
    gen_existing = make_existing_generator()
    try:
        ok_existing, failed_existing = run_mingus_on_solos(
            gen_existing, selected, INPUT_BARS, OUTPUT_BARS, GEN_SEED, log_prefix="existing ",
        )
    finally:
        gen_existing.close()

    print("\nrunning MINGUS (ours — paper-optimal, cond_pitch=D-C-B-BE-O / cond_duration=B-BE-O, epochs=10) ...",
          file=sys.stderr)
    gen_ours = make_ours_generator()
    try:
        ok_ours, failed_ours = run_mingus_on_solos(
            gen_ours, selected, INPUT_BARS, OUTPUT_BARS, GEN_SEED, log_prefix="ours     ",
        )
    finally:
        gen_ours.close()

    # Skip-and-keep-pair: оставляем только соло, где обе модели отработали успешно.
    surviving = [n for n in selected if n in ok_existing and n in ok_ours]
    dropped = sorted(set(selected) - set(surviving))
    if dropped:
        print(f"\ndropped {len(dropped)} solos (failed in one or both configs):", file=sys.stderr)
        for n in dropped:
            tag_e = "FAIL" if n in failed_existing else "ok"
            tag_o = "FAIL" if n in failed_ours else "ok"
            print(f"  {n} [existing={tag_e}, ours={tag_o}]", file=sys.stderr)
    print(f"\nfinal corpus size: {len(surviving)} solos", file=sys.stderr)

    print("\nloading real corpus (full WjazzD solos) ...", file=sys.stderr)
    real_pms = load_real_corpus(surviving)
    gen_pms_existing = [ok_existing[n] for n in surviving]
    gen_pms_ours = [ok_ours[n] for n in surviving]
    print(f"real={len(real_pms)} existing={len(gen_pms_existing)} ours={len(gen_pms_ours)}",
          file=sys.stderr)

    print("\ncomputing MGEval (real vs existing & real vs ours) ...", file=sys.stderr)
    rows = compute_mgeval(
        real_pms,
        {"existing": gen_pms_existing, "ours": gen_pms_ours},
    )
    kl_oa_existing = {
        r["feature"]: (r["kl"], r["oa"]) for r in rows if r["model"] == "existing"
    }
    kl_oa_ours = {
        r["feature"]: (r["kl"], r["oa"]) for r in rows if r["model"] == "ours"
    }

    out_path = Path(__file__).resolve().parent / "result.csv"
    write_step2_csv(out_path, kl_oa_existing, kl_oa_ours)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
