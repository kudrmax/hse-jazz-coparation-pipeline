"""Step 5 — 2×2 factorial (slicing × input source). Только MINGUS, наш чекпоинт.

4 клетки:
  1) full_wjazzd      — из step2b/result.csv (kl_ours, oa_ours).
  2) full_effendi     — НОВЫЙ прогон: real=40 whole solos vs gen_full.mid
                        ТОЛЬКО для 99 intersection-тем (Manifest filter:
                        status=ok AND not removed_from_corpus). 198 пьес.
  3) chunked_effendi  — из step4/result.csv (model=mingus). Group A:
                        kl_mean/oa_mean (single-shot), Group B:
                        bootstrap-mean.
  4) chunked_wjazzd   — НОВЫЙ прогон (детерминистский per seed=42):
                        15 selected solos → MINGUS-продолжения → 8-bar
                        chunks; Group A single-shot, Group B 20-bootstrap.

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/analytics/step5_factorial_2x2/run.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import music21 as m21
import numpy as np
import pretty_midi

COMP_ROOT = Path(__file__).resolve().parents[3]
ANALYTICS_ROOT = Path(__file__).resolve().parents[1]
if str(COMP_ROOT) not in sys.path:
    sys.path.insert(0, str(COMP_ROOT))
sys.path.insert(0, str(ANALYTICS_ROOT / "step1_baseline"))
sys.path.insert(0, str(ANALYTICS_ROOT / "step2_paper_reproduction"))

from _step1_common import load_full_solos  # noqa: E402
from _step2_common import (  # noqa: E402
    GEN_SEED,
    OUTPUT_BARS,
    make_ours_generator,
    run_mingus_on_solos,
    select_solo_names,
)
from manifest import Manifest  # noqa: E402
from mgeval.corpus_loader import (  # noqa: E402
    _extract_melody_part,
    _note_count,
    _score_chunk_to_pretty_midi,
    _split_into_bar_chunks,
)
from mgeval.features import FEATURES  # noqa: E402
from mgeval.pipeline import compute_mgeval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
XML_DIR = REPO_ROOT / "models" / "MINGUS" / "A_preprocessData" / "data" / "xml"
SLUG = "full-cleared-2samples"
SLUG_DIR = REPO_ROOT / "pipelines" / "comparation-pipeline" / "outputs" / SLUG

STEP2B_CSV = (
    ANALYTICS_ROOT / "step2_paper_reproduction" / "2b_long_seed" / "result.csv"
)
STEP4_CSV = ANALYTICS_ROOT / "step4_our_methodology" / "result.csv"

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


# === Loaders для готовых CSV ===

def _load_step2b_cell() -> dict[str, tuple[float, float]]:
    """Cell 1: feature → (kl_ours, oa_ours) из step2b."""
    out: dict[str, tuple[float, float]] = {}
    with STEP2B_CSV.open() as f:
        for row in csv.DictReader(f):
            out[row["feature"]] = (float(row["kl_ours"]), float(row["oa_ours"]))
    return out


def _load_step4_mingus_cell() -> dict[str, tuple[float, float]]:
    """Cell 3: feature → (kl_mean, oa_mean) для mingus из step4.

    В step4 Group A имеет n_trials=1, std=0, kl_min=kl_max=kl_mean —
    т.е. kl_mean это и есть single-shot значение. Group B имеет
    bootstrap-mean в kl_mean/oa_mean. Обе колонки именуются одинаково.
    """
    out: dict[str, tuple[float, float]] = {}
    with STEP4_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["model"] == "mingus":
                out[row["feature"]] = (float(row["kl_mean"]), float(row["oa_mean"]))
    return out


# === Cell 2: новый прогон на intersection 99 ===

def _intersection_themes() -> tuple[list[str], int]:
    """ТЕ ЖЕ темы, что compute_metrics.py:43-46 — intersection across models."""
    manifest = Manifest.load(SLUG_DIR / "manifest.json")
    active = [
        name for name, t in manifest.themes.items()
        if t.status == "ok" and not t.removed_from_corpus
    ]
    return active, int(manifest.samples_per_theme)


def _load_mingus_gen_full_intersection() -> tuple[list[pretty_midi.PrettyMIDI], int]:
    """gen_full.mid mingus только для 99 intersection-тем."""
    active, spt = _intersection_themes()
    pool: list[pretty_midi.PrettyMIDI] = []
    for theme in active:
        for i in range(spt):
            p = SLUG_DIR / "themes" / theme / "mingus" / f"sample_{i}" / "gen_full.mid"
            if p.is_file():
                pm = pretty_midi.PrettyMIDI(str(p))
                if sum(len(ins.notes) for ins in pm.instruments) > 0:
                    pool.append(pm)
    return pool, len(active)


def _cell2_full_effendi() -> dict[str, tuple[float, float]]:
    print("\n--- Cell 2: full_effendi (intersection 99 themes) ---", file=sys.stderr)
    real_pms = [pm for _, pm in load_full_solos()]
    print(f"  real: {len(real_pms)} whole WjazzD solos", file=sys.stderr)
    gen_pool, n_themes = _load_mingus_gen_full_intersection()
    print(
        f"  N_themes (Manifest intersection) = {n_themes}",
        file=sys.stderr,
    )
    print(
        f"  generated (mingus gen_full.mid, intersection): {len(gen_pool)} pieces",
        file=sys.stderr,
    )
    print("  computing MGEval (single-shot) ...", file=sys.stderr)
    rows = compute_mgeval(real_pms, {"mingus": gen_pool})
    return {r["feature"]: (r["kl"], r["oa"]) for r in rows}


# === Cell 4: chunked_wjazzd ===

def _load_real_chunks_for(selected: list[str], chunk_bars: int) -> list[pretty_midi.PrettyMIDI]:
    """m21-based нарезка для выбранных соло (использует приватные хелперы
    из mgeval.corpus_loader). Pickup отбрасывается, дегенеративные тоже."""
    chunks: list[pretty_midi.PrettyMIDI] = []
    for name in selected:
        xml = XML_DIR / f"{name}.xml"
        score = m21.converter.parse(str(xml))
        melody = _extract_melody_part(score)
        for c in _split_into_bar_chunks(melody, chunk_bars):
            pm = _score_chunk_to_pretty_midi(c)
            if _note_count(pm) > 0:
                chunks.append(pm)
    return chunks


def _slice_pm_into_bar_chunks(
    pm: pretty_midi.PrettyMIDI, chunk_bars: int
) -> list[pretty_midi.PrettyMIDI]:
    """Разрезать PrettyMIDI на непересекающиеся chunk_bars-окна по времени.
    Длина бара — по первому tempo event (fallback 120 BPM) + time_signature
    (fallback 4/4). Хвост короче chunk_bars — drop.
    """
    tempo_times, tempi = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) > 0 else 120.0
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        numer, denom = ts.numerator, ts.denominator
    else:
        numer, denom = 4, 4
    bar_sec = (60.0 / tempo) * numer * (4.0 / denom)
    win = chunk_bars * bar_sec

    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    if not notes:
        return []
    end = pm.get_end_time()
    chunks: list[pretty_midi.PrettyMIDI] = []
    t = 0.0
    while t + win <= end + 1e-6:
        chunk_pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        chunk_pm.time_signature_changes.append(
            pretty_midi.TimeSignature(numer, denom, 0.0)
        )
        ins_out = pretty_midi.Instrument(program=0)
        for n in notes:
            if t <= n.start < t + win:
                ins_out.notes.append(pretty_midi.Note(
                    pitch=n.pitch,
                    velocity=n.velocity,
                    start=n.start - t,
                    end=min(n.end, t + win) - t,
                ))
        if ins_out.notes:
            chunk_pm.instruments.append(ins_out)
            chunks.append(chunk_pm)
        t += win
    return chunks


def _cell4_chunked_wjazzd() -> dict[str, tuple[float, float]]:
    print("\n--- Cell 4: chunked_wjazzd (15 selected solos chunked) ---", file=sys.stderr)
    selected = select_solo_names()
    print(f"  selected {len(selected)} solos (random_seed=42)", file=sys.stderr)

    print(
        f"  running MINGUS (ours) input_bars=4 output_bars={OUTPUT_BARS} ...",
        file=sys.stderr,
    )
    gen = make_ours_generator()
    try:
        ok_gen, _failed = run_mingus_on_solos(
            gen, selected, input_bars=4, output_bars=OUTPUT_BARS, seed=GEN_SEED,
            log_prefix="ours ",
        )
    finally:
        gen.close()
    surviving = [n for n in selected if n in ok_gen]
    dropped = [n for n in selected if n not in surviving]
    if dropped:
        print(f"  dropped {len(dropped)} solos: {dropped}", file=sys.stderr)

    real_chunks = _load_real_chunks_for(surviving, CHUNK_BARS)
    gen_chunks: list[pretty_midi.PrettyMIDI] = []
    for name in surviving:
        gen_chunks.extend(_slice_pm_into_bar_chunks(ok_gen[name], CHUNK_BARS))
    print(
        f"  real_chunks={len(real_chunks)}, gen_chunks={len(gen_chunks)}",
        file=sys.stderr,
    )

    out: dict[str, tuple[float, float]] = {}

    # Group A — single-shot.
    print("  Group A: single-shot MGEval ...", file=sys.stderr)
    rows_a = compute_mgeval(real_chunks, {"mingus": gen_chunks})
    for r in rows_a:
        if r["feature"] in GROUP_A:
            out[r["feature"]] = (r["kl"], r["oa"])

    # Group B — 20 bootstrap.
    print(f"  Group B: {len(BOOTSTRAP_SEEDS)} bootstrap trials (N={N_SAMPLE}) ...",
          file=sys.stderr)
    acc: dict[str, dict[str, list[float]]] = {f: {"kl": [], "oa": []} for f in GROUP_B}
    for trial_idx, seed in enumerate(BOOTSTRAP_SEEDS, 1):
        rng = random.Random(seed)
        sampled = rng.sample(real_chunks, N_SAMPLE)
        rows = compute_mgeval(sampled, {"mingus": gen_chunks})
        for r in rows:
            if r["feature"] in GROUP_B:
                acc[r["feature"]]["kl"].append(r["kl"])
                acc[r["feature"]]["oa"].append(r["oa"])
        print(f"    [{trial_idx:2d}/{len(BOOTSTRAP_SEEDS)}] seed={seed}", file=sys.stderr)

    for feat in GROUP_B:
        kls = np.asarray(acc[feat]["kl"], dtype=float)
        oas = np.asarray(acc[feat]["oa"], dtype=float)
        out[feat] = (float(kls.mean()), float(oas.mean()))

    return out


def main() -> None:
    print(
        "=== step 5 — 2×2 factorial (slicing × input source) — MINGUS ours ===",
        file=sys.stderr,
    )
    print(f"slug_dir: {SLUG_DIR}", file=sys.stderr)

    print(f"\nloading cell 1 (full_wjazzd) from {STEP2B_CSV} ...", file=sys.stderr)
    cell1 = _load_step2b_cell()
    print(f"  cell 1: {len(cell1)} features", file=sys.stderr)

    print(f"\nloading cell 3 (chunked_effendi) from {STEP4_CSV} ...", file=sys.stderr)
    cell3 = _load_step4_mingus_cell()
    print(f"  cell 3: {len(cell3)} features", file=sys.stderr)

    cell2 = _cell2_full_effendi()
    cell4 = _cell4_chunked_wjazzd()

    out_path = Path(__file__).resolve().parent / "result.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "feature", "group",
            "full_wjazzd_kl", "full_wjazzd_oa",
            "full_effendi_kl", "full_effendi_oa",
            "chunked_effendi_kl", "chunked_effendi_oa",
            "chunked_wjazzd_kl", "chunked_wjazzd_oa",
        ])
        for feat in FEATURES.keys():
            group = "A" if feat in GROUP_A else "B"
            c1_kl, c1_oa = cell1[feat]
            c2_kl, c2_oa = cell2[feat]
            c3_kl, c3_oa = cell3[feat]
            c4_kl, c4_oa = cell4[feat]
            w.writerow([
                feat, group,
                f"{c1_kl:.6f}", f"{c1_oa:.6f}",
                f"{c2_kl:.6f}", f"{c2_oa:.6f}",
                f"{c3_kl:.6f}", f"{c3_oa:.6f}",
                f"{c4_kl:.6f}", f"{c4_oa:.6f}",
            ])
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
