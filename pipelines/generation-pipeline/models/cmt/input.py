from __future__ import annotations

from dataclasses import dataclass

from models.base.io import BaseGeneratorInput


@dataclass
class GeneratorCmtInput(BaseGeneratorInput):
    topk: int = 5
