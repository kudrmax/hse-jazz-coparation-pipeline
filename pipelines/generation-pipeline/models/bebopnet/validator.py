"""BebopnetInputValidator — extends CommonInputValidator with Bebop specifics."""
from __future__ import annotations

import music21 as m21

from models.base.io import BaseGeneratorInput
from models.base.validator import CommonInputValidator


class BebopnetInputValidator(CommonInputValidator):
    def validate(
        self, inp: BaseGeneratorInput, parsed_stream: m21.stream.Score
    ) -> None:
        super().validate(inp, parsed_stream)
        # TODO: check that the theme has >= 64 notes (Transformer-XL mem_len=64).
        # When implementing, verify the count matches how BebopNet's own
        # preprocessing computes head_len (see jazz_rnn/B_next_note_prediction/).
