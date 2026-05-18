"""CmtPostProcessor — extends CommonPostProcessor with CMT-specific work.

CMT runs inference in C major / A minor (the only keys in its training
distribution). Step 2 of generate() transposes the theme there; this
post-processor undoes that shift on the output MIDI so the user gets
their melody back in the theme's original key.

The semitone offset to undo is carried as `output.transpose_semitones`
(filled by GeneratorCmt._generate_impl). It already includes the sign
needed for transpose_midi_back, so we just pass it through.
"""
from __future__ import annotations

import music21 as m21

from models.base.io import BaseGeneratorInput, BaseGeneratorOutput
from models.base.post_processor import CommonPostProcessor

from .output import GeneratorCmtOutput
from .transposition import transpose_midi_back


class CmtPostProcessor(CommonPostProcessor):
    def process(
        self,
        inp: BaseGeneratorInput,
        parsed_stream: m21.stream.Score,
        output: BaseGeneratorOutput,
    ) -> None:
        super().process(inp, parsed_stream, output)
        assert isinstance(output, GeneratorCmtOutput)
        transpose_midi_back(output.midi, output.transpose_semitones)
