"""Single source of truth для списка моделей comparation-pipeline.

Используется в manifest, self_healer, metric_pipelines, phases.metrics
и backfill_musicxml — везде, где нужно итерировать по моделям в фиксированном
порядке.

Порядок (cmt → mingus → bebopnet) важен для CSV-колонок и значений в manifest.
"""
from __future__ import annotations

MODEL_NAMES: tuple[str, ...] = ("cmt", "mingus", "bebopnet")
