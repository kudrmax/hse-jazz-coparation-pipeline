"""Byte-identical guard для refactor шага 1.

Читает зафиксированный snapshot (sha256 + relative path) из fixtures/ и
сравнивает с текущим состоянием outputs/example/. Падает при первом
расхождении. После каждой задачи рефакторинга — must pass.

Исключения из baseline:
- `manifest.json` — содержит timestamps (`started_at`, `last_updated_at`,
  `samples[*].ts`). Проверяется логически отдельно после полного прогона.
- `theme_chunks/*.musicxml` — `music21.Score.write("musicxml")` пишет
  encoding-date (текущая дата) + random `score-instrument id`. theme.musicxml
  через `shutil.copy2` детерминирован, остаётся в baseline.
- `_metrics/*.csv` — стираются slow-тестом `test_smoke_example_slug`
  (он запускает `generate_batch --force` без compute_metrics). Метрики
  проверяются standalone-прогоном `compute_metrics.py --slug example`.

Тест НЕ запускает generate_batch.py — он только верифицирует, что
текущее состояние outputs/example/ совпадает с baseline'ом. Для
осмысленной проверки после рефакторинга нужно отдельно прогнать
`generate_batch.py --slug example` (Task 11).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_example_outputs_byte_identical():
    lines = (FIXTURES / "example_baseline_hashes.txt").read_text().splitlines()
    mismatches: list[str] = []
    missing: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        # Формат: "<sha256>  <relative-path>" (двойной пробел от shasum)
        h, rel_path = line.split("  ", 1)
        path = COMP_ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        actual = _sha256(path)
        if actual != h:
            mismatches.append(f"{rel_path}: expected {h[:12]}.., got {actual[:12]}..")
    assert not missing, f"missing files:\n" + "\n".join(missing)
    assert not mismatches, f"hash mismatches:\n" + "\n".join(mismatches)
