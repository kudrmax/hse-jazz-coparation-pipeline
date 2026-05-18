"""Tests for CommonInputValidator — checks shared by all three models."""
from pathlib import Path

import music21 as m21
import pytest

from models.base.io import BaseGeneratorInput
from models.base.validator import CommonInputValidator


def _build_n_bar_stream(n_bars: int) -> m21.stream.Score:
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        m.append(m21.note.Note("C5", quarterLength=4.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    return score


def test_rejects_non_positive_input_bars():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=0, output_bars=8
    )
    stream = _build_n_bar_stream(0)
    v = CommonInputValidator()
    with pytest.raises(ValueError, match="input_bars"):
        v.validate(inp, stream)


def test_rejects_non_positive_output_bars():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=0
    )
    stream = _build_n_bar_stream(8)
    v = CommonInputValidator()
    with pytest.raises(ValueError, match="output_bars"):
        v.validate(inp, stream)


def test_rejects_xml_shorter_than_input_bars():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(4)
    v = CommonInputValidator()
    with pytest.raises(ValueError, match="at least input_bars=8"):
        v.validate(inp, stream)


def test_accepts_xml_matching_input_bars():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(8)
    v = CommonInputValidator()
    v.validate(inp, stream)  # no exception


def test_accepts_xml_longer_than_input_bars():
    """Long themes are allowed — preprocessor will trim them to input_bars."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(32)
    v = CommonInputValidator()
    v.validate(inp, stream)  # no exception


def test_rejects_non_4_4_time_signature():
    """The three wrappers in the pipeline are 4/4-only — common check."""
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("3/4"))
    for bar in range(8):
        m = m21.stream.Measure(number=bar + 1)
        m.append(m21.note.Note("C5", quarterLength=3.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)

    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    v = CommonInputValidator()
    with pytest.raises(ValueError, match="4/4"):
        v.validate(inp, score)
