"""GeneratorBebopnet — pipeline-venv wrapper around the bebopnet-code fork.

__init__:
  1. Store parameters.
  2. Run _vocab_dump_runner.py once in bebopnet venv to fetch duration vocab
     for BebopnetPreprocessor (via legacy one-shot subprocess pattern).
  3. Construct PersistentSubprocessClient for inference (lazy spawn).

_generate_impl runs inference via PSClient (persistent subprocess).
"""
from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import pretty_midi

from models.base.generator import BaseGenerator
from models.base.io import BaseGeneratorInput
from models.base.persistent_subprocess_client import (
    PersistentSubprocessClient,
    SubprocessInferenceError,
)
# Legacy one-shot helper still used for vocab dump.
from models.base.subprocess_runner import run_subprocess_inference

from .input import GeneratorBebopnetInput
from .output import GeneratorBebopnetOutput
from .post_processor import BebopnetPostProcessor
from .preprocessor import BebopnetPreprocessor
from .validator import BebopnetInputValidator


class GeneratorBebopnet(BaseGenerator):
    def __init__(
        self,
        fork_root: Path,
        model_dir: Path,
        checkpoint: str = "model.pt",
        device: str = "cpu",
    ):
        self.fork_root = Path(fork_root)
        self.model_dir = Path(model_dir)
        self.checkpoint = checkpoint
        self.device = device
        self._validator = BebopnetInputValidator()
        self._post_processor = BebopnetPostProcessor()

        venv_python = self.fork_root / ".venv" / "bin" / "python"
        bbn_dir = Path(__file__).parent
        self._vocab_dump_script = bbn_dir / "_vocab_dump_runner.py"
        self._inference_script = bbn_dir / "_subprocess_runner.py"

        # Fetch duration vocab from forked venv (one-shot subprocess).
        vocab = self._fetch_vocab(venv_python)
        self._preprocessor = BebopnetPreprocessor(vocab=vocab)

        # Persistent inference subprocess — lazy spawn on first generate.
        self._client = PersistentSubprocessClient(
            venv_python=venv_python,
            runner_script=self._inference_script,
            config={
                "fork_root": str(self.fork_root),
                "model_dir": str(self.model_dir),
                "checkpoint": self.checkpoint,
                "device": self.device,
            },
        )

    def _fetch_vocab(self, venv_python: Path) -> frozenset[Fraction]:
        response = run_subprocess_inference(
            venv_python=venv_python,
            runner_script=self._vocab_dump_script,
            request={
                "fork_root": str(self.fork_root),
                "model_dir": str(self.model_dir),
            },
        )
        return frozenset(Fraction(num, den) for num, den in response["durations"])

    def _generate_impl(
        self,
        inp: BaseGeneratorInput,
        musicxml_path: Path,
    ) -> GeneratorBebopnetOutput:
        assert isinstance(inp, GeneratorBebopnetInput)

        with tempfile.TemporaryDirectory(prefix="bebopnet_gen_") as tmpdir:
            midi_path = Path(tmpdir) / "out.mid"
            request = {
                "musicxml_path": str(musicxml_path),
                "seed": inp.seed,
                "output_bars": inp.output_bars,
                "temperature": inp.temperature,
                "midi_out_path": str(midi_path),
            }
            response = self._client.request(request)
            if not response.get("ok"):
                raise SubprocessInferenceError(
                    f"BebopNet inference failed: {response.get('error', 'unknown')}",
                    stderr=response.get("traceback", ""),
                )
            midi_out = pretty_midi.PrettyMIDI(str(midi_path))

        return GeneratorBebopnetOutput(
            midi=midi_out,
            title=inp.get_musicxml_path().stem,
            seed=inp.seed,
            input_bars=inp.input_bars,
            output_bars=inp.output_bars,
            inference_time=0.0,
            temperature=inp.temperature,
            top_likelihood=float(response["top_likelihood"]),
        )

    def _close_impl(self) -> None:
        self._client.close()
