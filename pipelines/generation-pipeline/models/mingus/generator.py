"""GeneratorMingus — pipeline-venv wrapper around the MINGUS fork.

Один persistent forked subprocess на инстанс GeneratorMingus.
Spawn ленивый, shutdown через gen.close()/_close_impl().
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pretty_midi

from models.base.generator import BaseGenerator
from models.base.io import BaseGeneratorInput
from models.base.persistent_subprocess_client import (
    PersistentSubprocessClient,
    SubprocessInferenceError,
)

from .input import GeneratorMingusInput
from .output import GeneratorMingusOutput
from .post_processor import MingusPostProcessor
from .preprocessor import MingusPreprocessor
from .validator import MingusInputValidator


class GeneratorMingus(BaseGenerator):
    def __init__(
        self,
        fork_root: Path,
        data_path: Path,
        checkpoint_dir: Path,
        epochs: int = 100,
        cond: str = "I-C-NC-B-BE-O",
        cond_pitch: str | None = None,
        cond_duration: str | None = None,
        device: str = "cpu",
    ):
        self.fork_root = Path(fork_root)
        self.data_path = Path(data_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.epochs = epochs
        # `cond` остаётся для back-compat. Paper-optimal MINGUS использует
        # разное conditioning для pitch и duration (см. spec §17).
        self.cond_pitch = cond_pitch if cond_pitch is not None else cond
        self.cond_duration = cond_duration if cond_duration is not None else cond
        self.cond = cond
        self.device = device

        self._validator = MingusInputValidator()
        self._preprocessor = MingusPreprocessor()
        self._post_processor = MingusPostProcessor()

        self._client = PersistentSubprocessClient(
            venv_python=self.fork_root / ".venv" / "bin" / "python",
            runner_script=Path(__file__).parent / "_subprocess_runner.py",
            config={
                "fork_root": str(self.fork_root),
                "data_path": str(self.data_path),
                "checkpoint_dir": str(self.checkpoint_dir),
                "epochs": self.epochs,
                "cond_pitch": self.cond_pitch,
                "cond_duration": self.cond_duration,
                "device": self.device,
            },
        )

    def _generate_impl(
        self,
        inp: BaseGeneratorInput,
        musicxml_path: Path,
    ) -> GeneratorMingusOutput:
        assert isinstance(inp, GeneratorMingusInput)

        with tempfile.TemporaryDirectory(prefix="mingus_gen_") as tmpdir:
            midi_path = Path(tmpdir) / "out.mid"
            request = {
                "musicxml_path": str(musicxml_path),
                "seed": inp.seed,
                "input_bars": inp.input_bars,
                "output_bars": inp.output_bars,
                "temperature": inp.temperature,
                "midi_out_path": str(midi_path),
            }
            response = self._client.request(request)
            if not response.get("ok"):
                raise SubprocessInferenceError(
                    f"MINGUS inference failed: {response.get('error', 'unknown')}",
                    stderr=response.get("traceback", ""),
                )
            midi_out = pretty_midi.PrettyMIDI(str(midi_path))

        return GeneratorMingusOutput(
            midi=midi_out,
            title=str(response["title"]),
            seed=inp.seed,
            input_bars=inp.input_bars,
            output_bars=inp.output_bars,
            inference_time=0.0,
            tempo=float(response["tempo"]),
            temperature=inp.temperature,
            epochs=self.epochs,
            cond=self.cond,
        )

    def _close_impl(self) -> None:
        self._client.close()
