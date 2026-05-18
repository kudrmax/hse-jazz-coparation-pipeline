from __future__ import annotations

from dataclasses import dataclass

from models.base.io import BaseGeneratorOutput


@dataclass
class GeneratorMingusOutput(BaseGeneratorOutput):
    tempo: float = 0.0
    temperature: float = 1.0
    epochs: int = 0
    cond: str = ""
