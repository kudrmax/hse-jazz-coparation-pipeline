"""Финальная сводная таблица comparation-pipeline.

Главная точка входа — FinalTableStep. Внешние модули (phases.metrics)
импортируют только FinalTableStep.
"""
from __future__ import annotations

from .step import FinalTableStep

__all__ = ["FinalTableStep"]
