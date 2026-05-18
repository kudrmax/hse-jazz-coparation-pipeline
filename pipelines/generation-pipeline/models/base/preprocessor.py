"""Common preprocessing applied to every generator's parsed input stream.

Per-model subclasses inherit this class and call super().process(...)
before adding their own steps (Z-style vertical inheritance — same shape
as CommonInputValidator → *InputValidator and CommonPostProcessor →
*PostProcessor).

`process()` returns a (stream, path) pair so generators that consume the
xml stream (CMT) and generators that read the xml from disk (MINGUS,
BebopNet) both get a consistent, preprocessed view. If preprocessing is
a no-op, the original stream and path are returned unchanged. Otherwise
a temporary musicxml file is written and its path returned; the caller
(BaseGenerator.generate) is responsible for cleaning that file up after
generation finishes.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import music21 as m21

from .io import BaseGeneratorInput


class CommonPreprocessor:
    def process(
        self,
        inp: BaseGeneratorInput,
        parsed_stream: m21.stream.Score,
    ) -> tuple[m21.stream.Score, Path]:
        return self._trim_to_input_bars(inp, parsed_stream)

    def _trim_to_input_bars(
        self,
        inp: BaseGeneratorInput,
        parsed_stream: m21.stream.Score,
    ) -> tuple[m21.stream.Score, Path]:
        """If the parsed stream has more measures than `inp.input_bars`,
        return a trimmed copy of the first `input_bars` measures plus the
        path to a temporary musicxml file holding that copy. Otherwise
        return the original stream and path unchanged.

        m21.stream.Stream.measures(start, end) is the authoritative helper
        for slicing — it preserves TimeSignature, KeySignature and
        ChordSymbol elements anchored inside the kept measures.
        """
        measures = list(
            parsed_stream.recurse().getElementsByClass(m21.stream.Measure)
        )
        if len(measures) <= inp.input_bars:
            return parsed_stream, inp.musicxml_path

        trimmed = parsed_stream.measures(1, inp.input_bars)
        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False
        ) as f:
            tmp_path = Path(f.name)
        actual = Path(str(trimmed.write("musicxml", fp=str(tmp_path))))
        if actual != tmp_path:
            actual.replace(tmp_path)
        return trimmed, tmp_path
