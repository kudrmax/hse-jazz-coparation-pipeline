"""Common input validation shared by all generators.

Per-model validators inherit this class and call super().validate(...)
before adding their own checks (Z-style vertical inheritance).
"""
from __future__ import annotations

import music21 as m21

from .io import BaseGeneratorInput


REQUIRED_TIME_SIGNATURE = "4/4"


class CommonInputValidator:
    def validate(
        self, inp: BaseGeneratorInput, parsed_stream: m21.stream.Score
    ) -> None:
        self._check_bar_counts_positive(inp)
        self._check_xml_not_shorter_than_input_bars(inp, parsed_stream)
        self._check_time_signature_is_4_4(parsed_stream)
        # TODO: add musicxml correctness checks
        # (parses cleanly, has at least one Note, has at least one ChordSymbol).

    def _check_bar_counts_positive(self, inp: BaseGeneratorInput) -> None:
        if inp.input_bars <= 0:
            raise ValueError(
                f"input_bars must be positive, got {inp.input_bars}"
            )
        if inp.output_bars <= 0:
            raise ValueError(
                f"output_bars must be positive, got {inp.output_bars}"
            )

    def _check_xml_not_shorter_than_input_bars(
        self, inp: BaseGeneratorInput, parsed_stream: m21.stream.Score
    ) -> None:
        """Lower-bound check: xml must have at least `input_bars` measures.
        Longer themes are accepted; CommonPreprocessor trims them down to
        `input_bars` before generation. Shorter themes would silently lose
        notes inside model-specific tensor extraction (or worse — pad with
        ambiguous data), so we reject them up front."""
        measures = list(
            parsed_stream.recurse().getElementsByClass(m21.stream.Measure)
        )
        if len(measures) < inp.input_bars:
            raise ValueError(
                f"musicxml must contain at least input_bars={inp.input_bars} "
                f"measures; got {len(measures)}"
            )

    def _check_time_signature_is_4_4(
        self, parsed_stream: m21.stream.Score
    ) -> None:
        """All three wrappers are jazz/4-4 pipelines:
        CMT's quantization grid assumes 4/4; BebopNet's preprocessing
        explicitly checks numerator==4; MINGUS is trained on the same.
        """
        ts_elements = list(
            parsed_stream.recurse().getElementsByClass(m21.meter.TimeSignature)
        )
        if ts_elements and ts_elements[0].ratioString != REQUIRED_TIME_SIGNATURE:
            raise ValueError(
                f"musicxml must be in {REQUIRED_TIME_SIGNATURE}; "
                f"got {ts_elements[0].ratioString}"
            )
