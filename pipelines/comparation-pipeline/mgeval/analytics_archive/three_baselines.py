"""3 baseline-замера MGEval на 60 disjoint WJazzD xml.

Цель — измерить floor шума MGEval-pipeline'а при n=20 per side на разных
подвыборках real-данных. Используется как ground truth для интерпретации
итоговых KL/OA модель-vs-real в `outputs/<slug>/_metrics/mgeval.csv`:
числа в пределах baseline'а = модель статистически неразличима от real;
числа сильно выше KL / ниже OA = реальный сигнал отклонения.

baseline_1 = 20 random A vs 20 random B (внутри bucket 1, 40 xml)
baseline_2 = 20 random A vs 20 random B (внутри bucket 2, 40 xml)
baseline_3 = 20 random A vs 20 random B (внутри bucket 3, 40 xml)

3 bucket'а disjoint между собой (3 × 40 = 120 файлов выбраны из 850
доступных в models/MINGUS/A_preprocessData/data/xml/).
Внутри bucket — random split 20 vs 20 (disjoint by construction).

Запуск:
    pipelines/comparation-pipeline/.venv/bin/python \\
        pipelines/comparation-pipeline/mgeval/baselines/three_baselines.py

Output:
    pipelines/comparation-pipeline/mgeval/baselines/three_baselines.csv
Cache (ephemeral, регенерится):
    /tmp/mgeval_three_baselines_cache/<xml_stem>.mid
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]  # baselines/ → mgeval/ → comparation-pipeline/ → pipelines/ → repo
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"
sys.path.insert(0, str(COMP_ROOT))

import music21 as m21
import pretty_midi

from mgeval.pipeline import compute_mgeval

XML_DIR = REPO_ROOT / "models/MINGUS/A_preprocessData/data/xml"
OUT_CSV = THIS.parent / "three_baselines.csv"
CACHE_DIR = Path("/tmp/mgeval_three_baselines_cache")


def _extract_melody(score: m21.stream.Score) -> m21.stream.Part:
    for part in score.parts:
        if len(part.recurse().notes) > 0:
            return part
    raise ValueError("no part with notes")


def _xml_to_pm(xml_path: Path) -> pretty_midi.PrettyMIDI | None:
    """xml → midi через temp-файл → PrettyMIDI (с кэшированием на диск).

    Возвращает None на любой ошибке парсинга/конвертации (некоторые WJazzD-xml
    имеют extreme tempo/duration → corrupt midi → PrettyMIDI отказывается).
    """
    cache = CACHE_DIR / f"{xml_path.stem}.mid"
    try:
        if not cache.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            score = m21.converter.parse(str(xml_path))
            melody = _extract_melody(score)
            s = m21.stream.Score()
            s.append(melody)
            mf = m21.midi.translate.streamToMidiFile(s)
            with tempfile.NamedTemporaryFile(
                dir=CACHE_DIR, suffix=".mid", delete=False
            ) as tf:
                tmp = Path(tf.name)
            mf.open(str(tmp), "wb")
            mf.write()
            mf.close()
            tmp.replace(cache)
        return pretty_midi.PrettyMIDI(str(cache))
    except Exception as e:
        print(f"  skip {xml_path.name}: {e}", flush=True)
        cache.unlink(missing_ok=True)
        return None


def main() -> int:
    all_xml = sorted(XML_DIR.glob("*.xml"))
    print(f"available xml: {len(all_xml)}", flush=True)
    if len(all_xml) < 120:
        raise SystemExit(f"need >=120 xml, got {len(all_xml)}")

    # Random pick 120 disjoint, split в 3 bucket по 40.
    random.shuffle(all_xml)
    picked = all_xml[:120]
    buckets = [picked[0:40], picked[40:80], picked[80:120]]
    print("3 disjoint buckets of 40 each, total 120 unique files", flush=True)

    csv_rows: list[dict] = []
    for b_idx, bucket in enumerate(buckets, start=1):
        print(f"\n=== baseline_{b_idx}: loading 40 files ===", flush=True)
        pms = [_xml_to_pm(x) for x in bucket]
        pms = [
            p for p in pms
            if p is not None and sum(len(i.notes) for i in p.instruments) > 0
        ]
        if len(pms) < 40:
            print(f"  warn: dropped {40 - len(pms)} bad/empty files (using {len(pms)})", flush=True)
        if len(pms) < 4:
            raise SystemExit(f"bucket {b_idx} has only {len(pms)} valid files")
        random.shuffle(pms)
        half = len(pms) // 2
        A, B = pms[:half], pms[half:half * 2]
        print(f"  A={len(A)} B={len(B)} (disjoint by construction)", flush=True)
        rows = compute_mgeval(A, {"B": B})
        for r in rows:
            existing = next(
                (x for x in csv_rows if x["feature"] == r["feature"]), None
            )
            if existing is None:
                existing = {"feature": r["feature"]}
                csv_rows.append(existing)
            existing[f"baseline_{b_idx}_kl"] = f"{r['kl']:.6f}"
            existing[f"baseline_{b_idx}_oa"] = f"{r['oa']:.6f}"
        for r in rows:
            print(
                f"  {r['feature']:35s} kl={r['kl']:8.4f}  oa={r['oa']:6.4f}",
                flush=True,
            )

    fieldnames = [
        "feature",
        "baseline_1_kl", "baseline_1_oa",
        "baseline_2_kl", "baseline_2_oa",
        "baseline_3_kl", "baseline_3_oa",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
