from __future__ import annotations

from dataclasses import dataclass

from models.base.io import BaseGeneratorOutput


@dataclass
class GeneratorBebopnetOutput(BaseGeneratorOutput):
    temperature: float = 1.0
    top_likelihood: float = 0.0
