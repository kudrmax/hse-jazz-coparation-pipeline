"""MingusInputValidator — extends CommonInputValidator with Mingus specifics."""
from __future__ import annotations

import music21 as m21

from models.base.io import BaseGeneratorInput
from models.base.validator import CommonInputValidator


class MingusInputValidator(CommonInputValidator):
    def validate(
        self, inp: BaseGeneratorInput, parsed_stream: m21.stream.Score
    ) -> None:
        super().validate(inp, parsed_stream)

        if inp.output_bars % inp.input_bars != 0:
            raise ValueError(
                f"Mingus requires output_bars % input_bars == 0; "
                f"got input_bars={inp.input_bars}, output_bars={inp.output_bars}"
            )
