"""MINGUS continuation test: 15 random WjazzD xml → обрезка до первых N баров
(seed) → MINGUS-продолжение → MGEval(real vs generated).

Запускает MINGUS два раза:
- existing_checkpoint = их authorial (epochs=100, cond=I-C-NC-B-BE-O)
- our_checkpoint     = наш paper-optimal (epochs=10)

Все артефакты живут РЯДОМ со скриптом (под script-dir):
    inputs/<name>.xml          — обрезанные «seeds» (commit'ятся)
    real/<name>.mid            — full real соло (regenerated, в .gitignore)
    gen_existing/<name>.mid    — MINGUS authorial gen (в .gitignore)
    gen_ours/<name>.mid        — MINGUS наш gen (в .gitignore)
    mgeval_continuation.csv    — итог MGEval (commit'ится)

CLI:
    --input-bars N       (default 4) — сколько баров seed'а отрезаем
    --output-bars M      (default 12) — сколько баров просим MINGUS сгенерить
    --suffix STR         (опц.) — добавить суффикс к именам outputs и CSV
                                 (чтобы прогон с другой длиной не затирал предыдущий)
    --n-samples K        (default 15) — сколько random файлов взять

Примеры:
    # short seed (default):
    python run.py
    # long seed:
    python run.py --input-bars 16 --output-bars 16 --suffix _longseed
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT_DIR = THIS.parent
REPO_ROOT = THIS.parents[5]  # mingus_continuation/ → baselines/ → mgeval/ → comparation-pipeline/ → pipelines/ → repo

COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"
GEN_ROOT = REPO_ROOT / "pipelines/generation-pipeline"
sys.path.insert(0, str(COMP_ROOT))
sys.path.insert(0, str(GEN_ROOT))

import music21 as m21
import pretty_midi

from mgeval.pipeline import compute_mgeval
from models.mingus import GeneratorMingus
from models.mingus.input import GeneratorMingusInput

XML_DIR = REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml"
MINGUS_FORK = REPO_ROOT / "models/MINGUS"
MINGUS_DATA = MINGUS_FORK / "A_preprocessData/data/DATA.json"
CKPT_EXISTING = MINGUS_FORK / "B_train/models"
CKPT_OURS = MINGUS_FORK / "B_train/models/paper-optimal"

SEED = 42  # фиксировано: random выбор файлов одинаков между прогонами


def _extract_melody(score: m21.stream.Score) -> m21.stream.Part:
    for part in score.parts:
        if len(part.recurse().notes) > 0:
            return part
    raise ValueError("no part with notes")


def _truncate_xml_to_first_bars(xml_in: Path, xml_out: Path, n_bars: int) -> bool:
    """Обрезает xml до первых n_bars баров мелодии. Сохраняет под xml_out.
    Возвращает False если не получилось (corrupt/short).
    """
    try:
        score = m21.converter.parse(str(xml_in))
        new_score = m21.stream.Score()
        if score.metadata is not None:
            new_score.append(score.metadata)
        for part in score.parts:
            measures = list(part.getElementsByClass(m21.stream.Measure))
            full = [m for m in measures
                    if not (getattr(m, "paddingLeft", 0) > 0 or m.number == 0)]
            if len(full) < n_bars:
                return False
            new_part = m21.stream.Part()
            new_part.id = part.id
            if part.partName:
                new_part.partName = part.partName
            for m in full[:n_bars]:
                new_part.append(m)
            new_score.append(new_part)
        new_score.write("musicxml", fp=str(xml_out))
        return True
    except Exception as e:
        print(f"  truncate failed for {xml_in.name}: {e}", flush=True)
        return False


def _xml_to_pm(xml: Path) -> pretty_midi.PrettyMIDI | None:
    """Конвертирует полный xml → midi через temp-file → PrettyMIDI."""
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
        print(f"  xml→pm failed for {xml.name}: {e}", flush=True)
        return None


def pick_input_files(n: int, random_seed: int = SEED) -> list[Path]:
    """Случайно выбирает n xml файлов из 850 WjazzD. random_seed позволяет
    разные прогоны (например test-set bias check) брать разные подвыборки.
    """
    all_xml = sorted(XML_DIR.glob("*.xml"))
    rng = random.Random(random_seed)
    rng.shuffle(all_xml)
    return all_xml[:n * 3]


def prepare_inputs(
    n_samples: int, input_bars: int, inputs_dir: Path, real_dir: Path,
    random_seed: int = SEED,
) -> list[tuple[str, Path, Path]]:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    candidates = pick_input_files(n_samples, random_seed)
    out: list[tuple[str, Path, Path]] = []
    for xml in candidates:
        if len(out) >= n_samples:
            break
        name = xml.stem
        trunc = inputs_dir / f"{name}.xml"
        if not trunc.exists():
            if not _truncate_xml_to_first_bars(xml, trunc, input_bars):
                continue
        real_mid = real_dir / f"{name}.mid"
        if not real_mid.exists():
            real_pm = _xml_to_pm(xml)
            if real_pm is None:
                trunc.unlink(missing_ok=True)
                continue
            real_pm.write(str(real_mid))
        out.append((name, trunc, real_mid))
    if len(out) < n_samples:
        raise SystemExit(f"only {len(out)} valid inputs, wanted {n_samples}")
    return out


def run_mingus(
    gen: GeneratorMingus,
    inputs: list[tuple[str, Path, Path]],
    out_dir: Path,
    input_bars: int,
    output_bars: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, (name, trunc, _) in enumerate(inputs):
        midi_out = out_dir / f"{name}.mid"
        if midi_out.exists():
            out_paths.append(midi_out)
            continue
        print(f"  [{i+1}/{len(inputs)}] {name}", flush=True)
        try:
            inp = GeneratorMingusInput(
                musicxml_path=str(trunc),
                seed=42,
                input_bars=input_bars,
                output_bars=output_bars,
                temperature=1.0,
            )
            out = gen.generate(inp)
            out.midi.write(str(midi_out))
            out_paths.append(midi_out)
        except Exception as e:
            print(f"    skip {name}: {e}", flush=True)
    return out_paths


def load_pms(paths: list[Path]) -> list[pretty_midi.PrettyMIDI]:
    out: list[pretty_midi.PrettyMIDI] = []
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            if sum(len(i.notes) for i in pm.instruments) > 0:
                out.append(pm)
        except Exception as e:
            print(f"  load_pms skip {p.name}: {e}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-bars", type=int, default=4,
                    help="сколько баров seed'а отрезаем из real xml (default 4)")
    ap.add_argument("--output-bars", type=int, default=12,
                    help="сколько баров генерить (default 12)")
    ap.add_argument("--n-samples", type=int, default=15,
                    help="сколько random xml взять (default 15)")
    ap.add_argument("--suffix", type=str, default="",
                    help="суффикс к имени outputs и CSV (для разных прогонов)")
    args = ap.parse_args()

    print(f"script dir: {SCRIPT_DIR}", flush=True)
    print(f"input_bars={args.input_bars} output_bars={args.output_bars} "
          f"n_samples={args.n_samples} suffix={args.suffix!r}", flush=True)

    # Все пути относительно скрипта.
    inputs_dir = SCRIPT_DIR / f"inputs{args.suffix}"
    real_dir = SCRIPT_DIR / f"real{args.suffix}"
    gen_existing_dir = SCRIPT_DIR / f"gen_existing{args.suffix}"
    gen_ours_dir = SCRIPT_DIR / f"gen_ours{args.suffix}"
    out_csv = SCRIPT_DIR / f"mgeval_continuation{args.suffix}.csv"

    print(f"\n=== preparing inputs (truncate to first {args.input_bars} bars) ===",
          flush=True)
    inputs = prepare_inputs(
        args.n_samples, args.input_bars, inputs_dir, real_dir,
    )
    print(f"prepared {len(inputs)} inputs", flush=True)

    print("\n=== MINGUS existing_checkpoint (epochs=100, I-C-NC-B-BE-O) ===",
          flush=True)
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
            gen_existing, inputs, gen_existing_dir,
            args.input_bars, args.output_bars,
        )
    finally:
        gen_existing.close()

    print("\n=== MINGUS our_checkpoint (epochs=10, paper-optimal) ===",
          flush=True)
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
            gen_ours, inputs, gen_ours_dir,
            args.input_bars, args.output_bars,
        )
    finally:
        gen_ours.close()

    print("\n=== loading corpora ===", flush=True)
    real_pms = load_pms([t[2] for t in inputs])
    gen_existing_pms = load_pms(gen_existing_paths)
    gen_ours_pms = load_pms(gen_ours_paths)
    print(f"real={len(real_pms)} existing={len(gen_existing_pms)} "
          f"ours={len(gen_ours_pms)}", flush=True)

    print("\n=== computing MGEval ===", flush=True)
    rows = compute_mgeval(
        real_pms,
        {
            "existing_checkpoint": gen_existing_pms,
            "our_checkpoint": gen_ours_pms,
        },
    )

    csv_rows: dict[str, dict] = {}
    for r in rows:
        feat = r["feature"]
        if feat not in csv_rows:
            csv_rows[feat] = {"feature": feat}
        if r["model"] == "existing_checkpoint":
            csv_rows[feat]["kl_existing"] = f"{r['kl']:.6f}"
            csv_rows[feat]["oa_existing"] = f"{r['oa']:.6f}"
        else:
            csv_rows[feat]["kl_ours"] = f"{r['kl']:.6f}"
            csv_rows[feat]["oa_ours"] = f"{r['oa']:.6f}"

    fieldnames = ["feature", "kl_existing", "oa_existing", "kl_ours", "oa_ours"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for feat in csv_rows:
            w.writerow(csv_rows[feat])
    print(f"\nwrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
