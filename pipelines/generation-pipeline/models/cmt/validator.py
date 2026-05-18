"""CmtInputValidator — extends CommonInputValidator with CMT specifics."""
from __future__ import annotations

import music21 as m21

from models.base.io import BaseGeneratorInput
from models.base.validator import CommonInputValidator


class CmtInputValidator(CommonInputValidator):
    def __init__(self, num_bars: int):
        self.num_bars = num_bars

    def validate(
        self, inp: BaseGeneratorInput, parsed_stream: m21.stream.Score
    ) -> None:
        super().validate(inp, parsed_stream)

        if inp.input_bars != inp.output_bars:
            raise ValueError(
                f"CMT requires input_bars == output_bars; got "
                f"input_bars={inp.input_bars}, output_bars={inp.output_bars}"
            )
        if inp.output_bars * 2 != self.num_bars:
            raise ValueError(
                f"output_bars ({inp.output_bars}) must equal num_bars/2; "
                f"loaded checkpoint has num_bars={self.num_bars}, expected "
                f"output_bars={self.num_bars // 2}"
            )
