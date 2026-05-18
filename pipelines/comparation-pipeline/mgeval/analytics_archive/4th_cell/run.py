"""4th cell of 2×2 (slicing × input): chunked WjazzD-real vs chunked
MINGUS-generated-on-WjazzD-input.

3 другие точки уже есть:
- full-WjazzD: mingus_continuation/mgeval_continuation_longseed.csv
- full-effendi: mgeval_full_solos.csv
- chunked-effendi: mgeval_grouped.csv / mgeval.csv

Если эта 4-я точка по числам ≈ full-WjazzD → slicing не важен, input доминирует.
Если ≈ chunked-effendi → input не важен, slicing доминирует.

Метод:
1. 15 WjazzD-соло (seed=42, те же что в Эксп. 3 — inputs_longseed/).
2. Реюзим существующие MINGUS-генерации из gen_ours_longseed/.
3. Real (полные 15 соло) и gen (15 MINGUS-continuations) нарезаем на 8-bar
   chunks через postprocess.slice_midi.
4. MGEval(real_chunks vs gen_chunks). Один прогон + опциональный bootstrap
   для Group B (PC, PR, PI).

Output:
    mgeval_4th_cell.csv  — 9 строк (по одной на feature).
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT_DIR = THIS.parent
REPO_ROOT = THIS.parents[5]
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"
sys.path.insert(0, str(COMP_ROOT))

import music21 as m21
import numpy as np
import pretty_midi

from mgeval.distances import c_dist, kl_dist, overlap_area
from mgeval.features import FEATURES
from postprocess import slice_midi

# --- Конфигурация ---
MINGUS_CONT_DIR = (
    COMP_ROOT / "mgeval/baselines/mingus_continuation"
)
INPUTS_LONGSEED_DIR = MINGUS_CONT_DIR / "inputs_longseed"  # не используется,
# но именно эти имена использованы в gen_ours_longseed
GEN_OURS_LONGSEED_DIR = MINGUS_CONT_DIR / "gen_ours_longseed"
REAL_XML_DIR = REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml"
CHUNK_BARS = 8
OUT_CSV = SCRIPT_DIR / "mgeval_4th_cell.csv"

# Group B: bootstrap для нестабильных pitch-фич.
GROUP_B = ("total_used_pitch", "pitch_range", "avg_pitch_interval")
BOOTSTRAP_N_TRIALS = 20
BOOTSTRAP_SAMPLE_SIZE = 15  # N=15 random real_chunks


# --- Helpers ---


def _extract_melody(score: m21.stream.Score) -> m21.stream.Part:
    for part in score.parts:
        if len(part.recurse().notes) > 0:
            return part
    raise ValueError("no part with notes")


def _xml_to_pm(xml: Path) -> pretty_midi.PrettyMIDI | None:
    """Полный xml → PrettyMIDI через temp-file."""
    try:
        score = m21.converter.parse(str(xml))
        melody = _extract_melody(score)
        s = m21.stream.Score()
        s.append(melody)
        mf = m21.midi.translate.streamToMidiFile(s)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
            tmp = Path(tf.name)
        mf.open(str(tmp), "wb")
        mf.write()
        mf.close()
        pm = pretty_midi.PrettyMIDI(str(tmp))
        tmp.unlink(missing_ok=True)
        if sum(len(i.notes) for i in pm.instruments) == 0:
            return None
        return pm
    except Exception as e:
        print(f"  xml→pm failed {xml.name}: {e}", flush=True)
        return None


def _note_count(pm: pretty_midi.PrettyMIDI) -> int:
    return sum(len(i.notes) for i in pm.instruments)


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


def main() -> int:
    # 1) Имена 15 WjazzD-соло — берём из gen_ours_longseed/.
    gen_midi_paths = sorted(GEN_OURS_LONGSEED_DIR.glob("*.mid"))
    names = [p.stem for p in gen_midi_paths]
    print(f"using {len(names)} names from gen_ours_longseed:", flush=True)
    for n in names:
        print(f"  - {n}", flush=True)

    # 2) Real-корпус: full WjazzD соло → 8-bar chunks.
    print(f"\n=== loading & chunking real ({len(names)} wjazzd → 8-bar chunks) ===",
          flush=True)
    real_chunks: list[pretty_midi.PrettyMIDI] = []
    for name in names:
        xml = REAL_XML_DIR / f"{name}.xml"
        pm = _xml_to_pm(xml)
        if pm is None:
            print(f"  skip real {name}: failed to load", flush=True)
            continue
        sliced = slice_midi(pm, CHUNK_BARS)
        sliced_non_empty = [c for c in sliced if _note_count(c) > 0]
        print(f"  real {name}: {len(sliced)} chunks ({len(sliced_non_empty)} non-empty)",
              flush=True)
        real_chunks.extend(sliced_non_empty)
    print(f"real chunks total: {len(real_chunks)}", flush=True)

    # 3) Generated-корпус: gen_ours_longseed/*.mid → 8-bar chunks.
    print(f"\n=== chunking generated ({len(names)} gen → 8-bar chunks) ===",
          flush=True)
    gen_chunks: list[pretty_midi.PrettyMIDI] = []
    for p in gen_midi_paths:
        pm = pretty_midi.PrettyMIDI(str(p))
        sliced = slice_midi(pm, CHUNK_BARS)
        sliced_non_empty = [c for c in sliced if _note_count(c) > 0]
        print(f"  gen {p.stem}: {len(sliced)} chunks ({len(sliced_non_empty)} non-empty)",
              flush=True)
        gen_chunks.extend(sliced_non_empty)
    print(f"gen chunks total: {len(gen_chunks)}", flush=True)

    if len(real_chunks) < 2 or len(gen_chunks) < 1:
        raise SystemExit("too few chunks for MGEval")

    # 4) MGEval, single run, на 9 фичах.
    print("\n=== MGEval single run (chunked WjazzD vs chunked MINGUS-on-WjazzD) ===",
          flush=True)
    single_rows: list[dict] = []
    for feat_name, fn in FEATURES.items():
        kl, oa = _kl_oa_for_feature(real_chunks, gen_chunks, fn)
        single_rows.append({
            "feature": feat_name,
            "kl_chunked_wjazzd": kl,
            "oa_chunked_wjazzd": oa,
        })
        print(f"  {feat_name:35s} kl={kl:7.4f}  oa={oa:6.4f}", flush=True)

    # 5) Опциональный bootstrap для Group B (PC, PR, PI).
    print(f"\n=== bootstrap для Group B ({BOOTSTRAP_N_TRIALS} trials × N={BOOTSTRAP_SAMPLE_SIZE} real chunks) ===",
          flush=True)
    rng = np.random.default_rng(42)
    kl_acc: dict[str, list[float]] = {f: [] for f in GROUP_B}
    oa_acc: dict[str, list[float]] = {f: [] for f in GROUP_B}
    sample_size = min(BOOTSTRAP_SAMPLE_SIZE, len(real_chunks))
    for trial in range(BOOTSTRAP_N_TRIALS):
        idx = rng.choice(len(real_chunks), size=sample_size, replace=False)
        sample = [real_chunks[i] for i in idx]
        for feat in GROUP_B:
            kl, oa = _kl_oa_for_feature(sample, gen_chunks, FEATURES[feat])
            kl_acc[feat].append(kl)
            oa_acc[feat].append(oa)
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial+1}/{BOOTSTRAP_N_TRIALS} done", flush=True)

    # Добавляем bootstrap-колонки.
    bootstrap_by_feat: dict[str, dict] = {}
    for feat in GROUP_B:
        kls = np.array(kl_acc[feat])
        oas = np.array(oa_acc[feat])
        bootstrap_by_feat[feat] = {
            "kl_bootstrap_mean": float(kls.mean()),
            "kl_bootstrap_std": float(kls.std()),
            "oa_bootstrap_mean": float(oas.mean()),
            "oa_bootstrap_std": float(oas.std()),
        }
        print(f"  bootstrap {feat:25s} "
              f"kl={kls.mean():.4f}±{kls.std():.4f}  "
              f"oa={oas.mean():.4f}±{oas.std():.4f}",
              flush=True)

    # 6) CSV: 9 строк, single-run + bootstrap-колонки для Group B (пустые для Group A).
    fieldnames = [
        "feature",
        "kl_chunked_wjazzd", "oa_chunked_wjazzd",
        "kl_bootstrap_mean", "kl_bootstrap_std",
        "oa_bootstrap_mean", "oa_bootstrap_std",
        "n_real_chunks", "n_gen_chunks", "n_bootstrap_trials",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in single_rows:
            row = {
                "feature": r["feature"],
                "kl_chunked_wjazzd": f"{r['kl_chunked_wjazzd']:.6f}",
                "oa_chunked_wjazzd": f"{r['oa_chunked_wjazzd']:.6f}",
                "kl_bootstrap_mean": "",
                "kl_bootstrap_std": "",
                "oa_bootstrap_mean": "",
                "oa_bootstrap_std": "",
                "n_real_chunks": len(real_chunks),
                "n_gen_chunks": len(gen_chunks),
                "n_bootstrap_trials": "",
            }
            if r["feature"] in bootstrap_by_feat:
                b = bootstrap_by_feat[r["feature"]]
                row["kl_bootstrap_mean"] = f"{b['kl_bootstrap_mean']:.6f}"
                row["kl_bootstrap_std"] = f"{b['kl_bootstrap_std']:.6f}"
                row["oa_bootstrap_mean"] = f"{b['oa_bootstrap_mean']:.6f}"
                row["oa_bootstrap_std"] = f"{b['oa_bootstrap_std']:.6f}"
                row["n_bootstrap_trials"] = BOOTSTRAP_N_TRIALS
            w.writerow(row)
    print(f"\nwrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
