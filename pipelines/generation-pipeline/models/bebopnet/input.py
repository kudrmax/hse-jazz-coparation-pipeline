from __future__ import annotations

from dataclasses import dataclass

from models.base.io import BaseGeneratorInput


@dataclass
class GeneratorBebopnetInput(BaseGeneratorInput):
    temperature: float = 1.0
