"""Dataclasses для comparation-pipeline configuration.

Per-model sub-configs (CmtModelConfig, MingusModelConfig, BebopnetModelConfig)
описывают точно те же поля что gen-pipeline формат — derived_yaml.py
переписывает их в gen-pipeline формат при формировании tasks для run.py --batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SegmentationConfig:
    chunk_bars: int


@dataclass(frozen=True)
class CmtModelConfig:
    fork_root: Path
    hparams_yaml_path: Path
    checkpoint_path: Path
    topk: int
    input_bars: int
    output_bars: int


@dataclass(frozen=True)
class MingusModelConfig:
    fork_root: Path
    data_path: Path
    checkpoint_dir: Path
    epochs: int
    cond_pitch: str
    cond_duration: str
    temperature: float
    input_bars: int | Literal["auto"]
    output_bars: int | Literal["auto"]


@dataclass(frozen=True)
class BebopnetModelConfig:
    fork_root: Path
    model_dir: Path
    checkpoint: str
    temperature: float
    input_bars: int | Literal["auto"]
    output_bars: int | Literal["auto"]


@dataclass(frozen=True)
class ComparationConfig:
    slug: str
    samples_per_theme: int
    device: str
    themes_limit: int | Literal["all"]
    output_formats: tuple[str, ...]
    segmentation: SegmentationConfig
    cmt: CmtModelConfig
    mingus: MingusModelConfig
    bebopnet: BebopnetModelConfig
