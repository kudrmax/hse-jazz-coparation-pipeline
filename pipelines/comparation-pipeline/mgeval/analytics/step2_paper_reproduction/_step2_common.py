"""Общие хелперы для step2_paper_reproduction.

- `select_solos`: 15 имён test-соло, отобранных с random_seed=42.
- `load_real_corpus`: их же → list[PrettyMIDI] целых соло.
- `make_existing_generator` / `make_ours_generator`: фабрики GeneratorMingus
  для двух конфигураций чекпоинта.
- `run_mingus_on_solos`: прогнать генерацию (input_bars, output_bars) на
  списке xml → list[PrettyMIDI].
- `write_step2_csv`: запись CSV в требуемом формате (feature,
  kl_existing, oa_existing, kl_ours, oa_ours, .6f).

MGEval — из mgeval/ (наша реализация, никаких сторонних импортов).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[5]
COMP_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
GEN_PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"

for _p in (COMP_PIPELINE_ROOT, GEN_PIPELINE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Shared loader из step1 для целых соло.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step1_baseline"))

from _step1_common import load_full_solos  # noqa: E402
from mgeval.features import FEATURES  # noqa: E402
from models.base.persistent_subprocess_client import SubprocessInferenceError  # noqa: E402
from models.mingus import GeneratorMingus, GeneratorMingusInput  # noqa: E402

SPLIT_JSON = REPO_ROOT / "pipelines" / "training-pipeline" / "wjazzd_split.json"
XML_DIR = REPO_ROOT / "models" / "MINGUS" / "A_preprocessData" / "data" / "xml"
FORK_ROOT = REPO_ROOT / "models" / "MINGUS"

SELECT_SEED = 42
GEN_SEED = 42
N_SOLOS = 15
OUTPUT_BARS = 12


def select_solo_names(seed: int = SELECT_SEED, n: int = N_SOLOS) -> list[str]:
    """С seed-ом отобрать `n` имён из split.json[test]. Возвращает в
    отсортированном порядке (детерминизм между подшагами и запусками).
    """
    test_names = sorted(json.loads(SPLIT_JSON.read_text())["test"])
    rng = random.Random(seed)
    chosen = rng.sample(test_names, n)
    return sorted(chosen)


def load_real_corpus(selected: list[str]) -> list[pretty_midi.PrettyMIDI]:
    """Целые соло как list[PrettyMIDI] (pickup отбрасывается, дегенеративные — тоже)."""
    all_solos = load_full_solos(SPLIT_JSON, XML_DIR)
    name_to_pm = dict(all_solos)
    out: list[pretty_midi.PrettyMIDI] = []
    for name in selected:
        if name in name_to_pm:
            out.append(name_to_pm[name])
    return out


def make_existing_generator(device: str = "cpu") -> GeneratorMingus:
    """Авторский paper-MINGUS: cond=I-C-NC-B-BE-O, epochs=100."""
    return GeneratorMingus(
        fork_root=FORK_ROOT,
        data_path=FORK_ROOT / "A_preprocessData" / "data" / "DATA.json",
        checkpoint_dir=FORK_ROOT / "B_train" / "models",
        epochs=100,
        cond="I-C-NC-B-BE-O",
        device=device,
    )


def make_ours_generator(device: str = "cpu") -> GeneratorMingus:
    """Наш paper-optimal: pitch=D-C-B-BE-O, duration=B-BE-O, epochs=10."""
    return GeneratorMingus(
        fork_root=FORK_ROOT,
        data_path=FORK_ROOT / "A_preprocessData" / "data" / "DATA.json",
        checkpoint_dir=FORK_ROOT / "B_train" / "models" / "paper-optimal",
        epochs=10,
        cond_pitch="D-C-B-BE-O",
        cond_duration="B-BE-O",
        device=device,
    )


def run_mingus_on_solos(
    generator: GeneratorMingus,
    selected_names: list[str],
    input_bars: int,
    output_bars: int = OUTPUT_BARS,
    seed: int = GEN_SEED,
    log_prefix: str = "",
) -> tuple[dict[str, pretty_midi.PrettyMIDI], list[str]]:
    """Прогнать `generator` на каждом xml из selected_names.

    Возвращает (name → PrettyMIDI) для успешных и список упавших имён.
    MINGUS xmlToStructuredSong иногда падает с IndexError при коротком
    trim (pickup без аккорда в первом такте). При падении соло
    помечается как failed — вызывающий код должен синхронно выкинуть
    его и из real, и из остальных моделей, чтобы корпуса оставались
    парными.
    """
    ok: dict[str, pretty_midi.PrettyMIDI] = {}
    failed: list[str] = []
    for i, name in enumerate(selected_names, 1):
        xml = XML_DIR / f"{name}.xml"
        try:
            out = generator.generate(GeneratorMingusInput(
                musicxml_path=xml,
                seed=seed,
                input_bars=input_bars,
                output_bars=output_bars,
            ))
        except Exception as e:
            # SubprocessInferenceError — падение внутри MINGUS-форка
            # (e.g. xmlToStructuredSong на pickup без аккорда).
            # m21.analysis.DiscreteAnalysisException — m21.analyze('key')
            # в CommonPostProcessor на сложных тональностях.
            # Ловим всё, чтобы trial не падал — соло выкидывается из
            # пары real/gen синхронно.
            failed.append(name)
            print(
                f"  {log_prefix}[{i:2d}/{len(selected_names)}] {name}: FAILED "
                f"({type(e).__name__}: {e})",
                file=sys.stderr,
            )
            continue
        pm = out.get_midi()
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        print(
            f"  {log_prefix}[{i:2d}/{len(selected_names)}] {name}: notes={n_notes}",
            file=sys.stderr,
        )
        ok[name] = pm
    return ok, failed


def write_step2_csv(
    csv_path: Path,
    kl_oa_existing: dict[str, tuple[float, float]],
    kl_oa_ours: dict[str, tuple[float, float]],
) -> None:
    """CSV строго по ТЗ:
    feature, kl_existing, oa_existing, kl_ours, oa_ours
    Порядок строк = порядок в FEATURES registry (9 фич). Точность 6 знаков.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "kl_existing", "oa_existing", "kl_ours", "oa_ours"])
        for feat in FEATURES.keys():
            kl_e, oa_e = kl_oa_existing[feat]
            kl_o, oa_o = kl_oa_ours[feat]
            w.writerow([
                feat,
                f"{kl_e:.6f}", f"{oa_e:.6f}",
                f"{kl_o:.6f}", f"{oa_o:.6f}",
            ])
