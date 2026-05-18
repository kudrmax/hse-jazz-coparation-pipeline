"""GeneratorCmt — pipeline-venv wrapper around the CMT-pytorch fork.

Один persistent forked subprocess на инстанс GeneratorCmt. Spawn ленивый
(на первом _generate_impl), shutdown через gen.close()/_close_impl().
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

from .input import GeneratorCmtInput
from .output import GeneratorCmtOutput
from .post_processor import CmtPostProcessor
from .preprocessor import CmtPreprocessor
from .validator import CmtInputValidator


class GeneratorCmt(BaseGenerator):
    def __init__(
        self,
        fork_root: Path,
        hparams_yaml_path: Path,
        checkpoint_path: Path,
        device: str = "cpu",
    ):
        self.fork_root = Path(fork_root)
        self.hparams_yaml_path = Path(hparams_yaml_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Read hparams for validator (num_bars).
        import yaml
        with open(self.hparams_yaml_path) as f:
            hparams = yaml.safe_load(f)
        self.num_bars = int(hparams["model"]["num_bars"])

        self._validator = CmtInputValidator(num_bars=self.num_bars)
        self._preprocessor = CmtPreprocessor()
        self._post_processor = CmtPostProcessor()

        self._client = PersistentSubprocessClient(
            venv_python=self.fork_root / ".venv" / "bin" / "python",
            runner_script=Path(__file__).parent / "_subprocess_runner.py",
            config={
                "fork_root": str(self.fork_root),
                "hparams_yaml_path": str(self.hparams_yaml_path),
                "checkpoint_path": str(self.checkpoint_path),
                "device": self.device,
            },
        )

    def _generate_impl(
        self,
        inp: BaseGeneratorInput,
        musicxml_path: Path,
    ) -> GeneratorCmtOutput:
        assert isinstance(inp, GeneratorCmtInput)

        with tempfile.TemporaryDirectory(prefix="cmt_gen_") as tmpdir:
            midi_path = Path(tmpdir) / "out.mid"
            request = {
                "musicxml_path": str(musicxml_path),
                "seed": inp.seed,
                "input_bars": inp.input_bars,
                "output_bars": inp.output_bars,
                "topk": inp.topk,
                "midi_out_path": str(midi_path),
            }
            response = self._client.request(request)
            if not response.get("ok"):
                raise SubprocessInferenceError(
                    f"CMT inference failed: {response.get('error', 'unknown')}",
                    stderr=response.get("traceback", ""),
                )
            midi_out = pretty_midi.PrettyMIDI(str(midi_path))

        return GeneratorCmtOutput(
            midi=midi_out,
            title=inp.get_musicxml_path().stem,
            seed=inp.seed,
            input_bars=inp.input_bars,
            output_bars=inp.output_bars,
            inference_time=0.0,
            num_bars=int(response["num_bars"]),
            frame_per_bar=int(response["frame_per_bar"]),
            topk=int(response["topk"]),
            checkpoint_epoch=response["checkpoint_epoch"],
            transpose_semitones=int(response["transpose_semitones"]),
        )

    def _close_impl(self) -> None:
        self._client.close()
