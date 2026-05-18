from __future__ import annotations

from dataclasses import dataclass

from models.base.io import BaseGeneratorOutput


@dataclass
class GeneratorCmtOutput(BaseGeneratorOutput):
    num_bars: int = 0
    frame_per_bar: int = 0
    topk: int = 0
    checkpoint_epoch: int | None = None
    # Carries the signed semitone offset CmtPostProcessor must apply to
    # bring the output MIDI back into the theme's original key. Filled by
    # GeneratorCmt._generate_impl; consumed by CmtPostProcessor.process.
    transpose_semitones: int = 0
