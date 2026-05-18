"""BaseGenerator — abstract orchestrator with Template Method.

Concrete wrappers (GeneratorMingus, GeneratorBebopnet, GeneratorCmt)
override _generate_impl(...) only. The public generate(...) handles
parsing, validation, preprocessing, post-processing, and timing in one
place.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import music21 as m21

from .io import BaseGeneratorInput, BaseGeneratorOutput
from .post_processor import CommonPostProcessor
from .preprocessor import CommonPreprocessor
from .validator import CommonInputValidator


class BaseGenerator(ABC):
    _validator: CommonInputValidator
    _preprocessor: CommonPreprocessor
    _post_processor: CommonPostProcessor

    def generate(self, inp: BaseGeneratorInput) -> BaseGeneratorOutput:
        parsed_stream = m21.converter.parse(str(inp.get_musicxml_path()))
        self._validator.validate(inp, parsed_stream)

        preprocessed_stream, preprocessed_path = self._preprocessor.process(
            inp, parsed_stream
        )
        tmp_to_cleanup: list[Path] = []
        if preprocessed_path != inp.get_musicxml_path():
            tmp_to_cleanup.append(preprocessed_path)

        try:
            t0 = time.perf_counter()
            out = self._generate_impl(inp, preprocessed_path)
            self._post_processor.process(inp, preprocessed_stream, out)
            out.inference_time = time.perf_counter() - t0
        finally:
            for p in tmp_to_cleanup:
                p.unlink(missing_ok=True)
        return out

    def close(self) -> None:
        """Release any persistent resources (e.g., forked-venv subprocess).
        Subclasses with persistent state override _close_impl(). Default is noop.
        Idempotent — safe to call multiple times.
        """
        if hasattr(self, "_close_impl"):
            self._close_impl()

    @abstractmethod
    def _generate_impl(
        self,
        inp: BaseGeneratorInput,
        musicxml_path: Path,
    ) -> BaseGeneratorOutput:
        ...
