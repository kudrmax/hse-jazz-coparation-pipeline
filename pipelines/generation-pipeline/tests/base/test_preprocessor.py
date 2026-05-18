"""Tests for CommonPreprocessor — trim-to-input_bars + tmp musicxml file."""
from pathlib import Path

import music21 as m21
import pytest

from models.base.io import BaseGeneratorInput
from models.base.preprocessor import CommonPreprocessor


def _build_n_bar_stream(n_bars: int, with_chords: bool = False) -> m21.stream.Score:
    part = m21.stream.Part()
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        if bar == 0:
            # TimeSignature / Key live inside the first measure — that's
            # how m21 lays them out when parsing real musicxml, and that's
            # where stream.measures(start, end) expects them so it can
            # carry them onto a sliced stream.
            m.insert(0.0, m21.meter.TimeSignature("4/4"))
            m.insert(0.0, m21.key.Key("C"))
        m.append(m21.note.Note("C5", quarterLength=4.0))
        if with_chords:
            m.insert(0.0, m21.harmony.ChordSymbol("Cmaj7"))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    return score


def test_no_trim_when_xml_matches_input_bars():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.musicxml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(8)
    pre = CommonPreprocessor()

    new_stream, new_path = pre.process(inp, stream)

    assert new_stream is stream
    assert new_path == inp.musicxml_path  # original path returned unchanged


def test_no_trim_when_xml_shorter_than_input_bars():
    """Validator guards the < case before preprocessor sees it, but we
    still must not pretend to trim — return inputs as-is."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.musicxml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(4)
    pre = CommonPreprocessor()

    new_stream, new_path = pre.process(inp, stream)

    assert new_stream is stream
    assert new_path == inp.musicxml_path


def test_trims_when_xml_longer_than_input_bars(tmp_path):
    inp = BaseGeneratorInput(
        musicxml_path=tmp_path / "src.musicxml", seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(32)
    pre = CommonPreprocessor()

    new_stream, new_path = pre.process(inp, stream)

    new_measures = list(
        new_stream.recurse().getElementsByClass(m21.stream.Measure)
    )
    assert len(new_measures) == 8
    assert new_path != inp.musicxml_path
    assert new_path.is_file()
    assert new_path.suffix == ".musicxml"


def test_trim_preserves_time_signature_key_and_chord_symbols(tmp_path):
    inp = BaseGeneratorInput(
        musicxml_path=tmp_path / "src.musicxml", seed=1, input_bars=4, output_bars=4
    )
    stream = _build_n_bar_stream(16, with_chords=True)
    pre = CommonPreprocessor()

    new_stream, _ = pre.process(inp, stream)

    ts_elements = list(
        new_stream.recurse().getElementsByClass(m21.meter.TimeSignature)
    )
    keys = list(new_stream.recurse().getElementsByClass(m21.key.Key))
    chord_symbols = list(
        new_stream.recurse().getElementsByClass(m21.harmony.ChordSymbol)
    )

    assert ts_elements, "TimeSignature must survive trim"
    assert ts_elements[0].ratioString == "4/4"
    assert keys, "Key must survive trim"
    assert len(chord_symbols) == 4, (
        f"Each of 4 trimmed measures had a ChordSymbol; got {len(chord_symbols)}"
    )


def test_trimmed_path_is_writable_musicxml_round_trip(tmp_path):
    """The tmp musicxml file written by trim must parse back through m21
    and yield the same measure count."""
    inp = BaseGeneratorInput(
        musicxml_path=tmp_path / "src.musicxml", seed=1, input_bars=8, output_bars=8
    )
    stream = _build_n_bar_stream(32)
    pre = CommonPreprocessor()

    _, new_path = pre.process(inp, stream)

    parsed_back = m21.converter.parse(str(new_path))
    measures = list(
        parsed_back.recurse().getElementsByClass(m21.stream.Measure)
    )
    assert len(measures) == 8
