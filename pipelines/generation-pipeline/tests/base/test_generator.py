"""Tests for BaseGenerator's Template Method via a fake concrete subclass."""
from pathlib import Path
from unittest.mock import MagicMock

import music21 as m21
import pretty_midi
import pytest

from models.base.generator import BaseGenerator
from models.base.io import BaseGeneratorInput, BaseGeneratorOutput
from models.base.post_processor import CommonPostProcessor
from models.base.preprocessor import CommonPreprocessor
from models.base.validator import CommonInputValidator


def _build_n_bar_xml(tmp_path: Path, n_bars: int) -> Path:
    part = m21.stream.Part()
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        if bar == 0:
            m.insert(0.0, m21.meter.TimeSignature("4/4"))
        m.append(m21.note.Note("C5", quarterLength=4.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    path = tmp_path / f"theme_{n_bars}bar.musicxml"
    score.write("musicxml", fp=str(path))
    return path


class _FakeGenerator(BaseGenerator):
    """Minimal subclass used only by these tests."""

    def __init__(self):
        self._validator = CommonInputValidator()
        self._preprocessor = CommonPreprocessor()
        self._post_processor = CommonPostProcessor()
        self.received_path: Path | None = None
        self.received_measure_count: int | None = None

    def _generate_impl(
        self,
        inp: BaseGeneratorInput,
        musicxml_path: Path,
    ) -> BaseGeneratorOutput:
        self.received_path = musicxml_path
        parsed = m21.converter.parse(str(musicxml_path))
        self.received_measure_count = len(
            list(parsed.recurse().getElementsByClass(m21.stream.Measure))
        )
        return BaseGeneratorOutput(
            midi=pretty_midi.PrettyMIDI(),
            title="fake",
            seed=inp.seed,
            input_bars=inp.input_bars,
            output_bars=inp.output_bars,
            inference_time=-1.0,  # generate() must overwrite
        )


def test_cannot_instantiate_base_generator_directly():
    with pytest.raises(TypeError):
        BaseGenerator()


def test_generate_calls_template_method_in_order(tmp_path):
    xml = _build_n_bar_xml(tmp_path, 8)
    inp = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=8, output_bars=8
    )
    gen = _FakeGenerator()

    spy_validator = MagicMock(wraps=gen._validator.validate)
    gen._validator.validate = spy_validator
    spy_preprocessor = MagicMock(wraps=gen._preprocessor.process)
    gen._preprocessor.process = spy_preprocessor

    out = gen.generate(inp)

    spy_validator.assert_called_once()
    spy_preprocessor.assert_called_once()
    assert isinstance(out, BaseGeneratorOutput)
    assert out.title == "fake"


def test_generate_overwrites_inference_time(tmp_path):
    xml = _build_n_bar_xml(tmp_path, 8)
    inp = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=8, output_bars=8
    )
    gen = _FakeGenerator()
    out = gen.generate(inp)
    assert out.inference_time >= 0.0
    assert out.inference_time != -1.0


def test_validation_failure_propagates(tmp_path):
    xml = _build_n_bar_xml(tmp_path, 8)
    bad = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=16, output_bars=8
    )
    gen = _FakeGenerator()
    with pytest.raises(ValueError, match="at least input_bars=16"):
        gen.generate(bad)


def test_preprocessor_runs_between_validate_and_generate_impl(tmp_path):
    """Order: validate → preprocess → _generate_impl. The fake records
    what _generate_impl sees; we assert preprocess actually ran by
    confirming _generate_impl received the trimmed view."""
    xml = _build_n_bar_xml(tmp_path, 32)
    inp = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=8, output_bars=8
    )
    gen = _FakeGenerator()

    call_order: list[str] = []
    orig_validate = gen._validator.validate
    orig_process = gen._preprocessor.process

    def track_validate(*args, **kwargs):
        call_order.append("validate")
        return orig_validate(*args, **kwargs)

    def track_process(*args, **kwargs):
        call_order.append("preprocess")
        return orig_process(*args, **kwargs)

    gen._validator.validate = track_validate
    gen._preprocessor.process = track_process

    gen.generate(inp)
    call_order.append("after-generate-impl")  # appended after generate_impl runs

    assert call_order[0] == "validate"
    assert call_order[1] == "preprocess"
    assert gen.received_measure_count == 8, (
        f"_generate_impl saw {gen.received_measure_count} measures; "
        "preprocessor must trim 32 → 8"
    )
    assert gen.received_path != xml, (
        "_generate_impl must receive the preprocessed tmp path, not the original"
    )


def test_preprocessor_no_op_passes_original_path(tmp_path):
    """When xml already matches input_bars, the original path flows
    through to _generate_impl untouched."""
    xml = _build_n_bar_xml(tmp_path, 8)
    inp = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=8, output_bars=8
    )
    gen = _FakeGenerator()
    gen.generate(inp)
    assert gen.received_path == xml


def test_preprocessor_tmp_file_is_cleaned_up_after_generate(tmp_path):
    xml = _build_n_bar_xml(tmp_path, 32)
    inp = BaseGeneratorInput(
        musicxml_path=xml, seed=1, input_bars=8, output_bars=8
    )
    gen = _FakeGenerator()
    gen.generate(inp)

    # The tmp file recorded on the fake must no longer exist.
    assert gen.received_path is not None
    assert not gen.received_path.exists(), (
        f"tmp musicxml {gen.received_path} should have been deleted by "
        "BaseGenerator.generate()'s try/finally"
    )


def test_close_delegates_to_close_impl(tmp_path):
    from models.base.generator import BaseGenerator
    from models.base.io import BaseGeneratorInput, BaseGeneratorOutput
    from models.base.preprocessor import CommonPreprocessor
    from models.base.post_processor import CommonPostProcessor
    from models.base.validator import CommonInputValidator

    closed = []

    class StubGen(BaseGenerator):
        def __init__(self):
            self._validator = CommonInputValidator()
            self._preprocessor = CommonPreprocessor()
            self._post_processor = CommonPostProcessor()

        def _generate_impl(self, inp, musicxml_path):
            raise NotImplementedError

        def _close_impl(self):
            closed.append(True)

    gen = StubGen()
    gen.close()
    assert closed == [True]


def test_close_default_is_noop():
    from models.base.generator import BaseGenerator
    from models.base.preprocessor import CommonPreprocessor
    from models.base.post_processor import CommonPostProcessor
    from models.base.validator import CommonInputValidator

    class StubGen(BaseGenerator):
        def __init__(self):
            self._validator = CommonInputValidator()
            self._preprocessor = CommonPreprocessor()
            self._post_processor = CommonPostProcessor()

        def _generate_impl(self, inp, musicxml_path):
            raise NotImplementedError

    gen = StubGen()
    gen.close()  # должно не бросить — нет _close_impl override
