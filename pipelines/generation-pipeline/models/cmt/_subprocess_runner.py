# pipelines/generation-pipeline/models/cmt/_subprocess_runner.py
"""CMT inference subprocess runner — server-mode in models/CMT-pytorch/.venv.

Protocol:
  argv:        --server
  first stdin: JSON config (fork_root, hparams_yaml_path, checkpoint_path, device)
  loop stdin:  JSON request per line (musicxml_path, seed, input_bars,
               output_bars, topk, midi_out_path)
  loop stdout: JSON response per line ({"ok": true, "transpose_semitones": ..., ...}
               or {"ok": false, "error": "..."})
  EOF on stdin → graceful exit.

Checkpoint грузится один раз при первом запросе (lazy, чтобы setup-fail
проявлялся как fail-ответ на запрос).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import music21 as m21
import pretty_midi
import torch
import yaml


def _add_pipeline_root_to_path() -> None:
    pipeline_root = Path(__file__).resolve().parents[2]
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))


class _CmtServer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.checkpoint_epoch = None
        self.num_bars = None
        self.frame_per_bar = None
        self.pitch_range = None
        self.device = None
        self._ChordConditionedMelodyTransformer = None
        self._analyze_key = None
        self._transpose_to_target = None
        self._xml_to_tensors = None
        self._pitch_to_midi = None
        self._MELODY_INSTRUMENT_INDEX = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        sys.path.insert(0, str(Path(self.config["fork_root"])))
        from model import ChordConditionedMelodyTransformer  # noqa
        from models.cmt.constants import MELODY_INSTRUMENT_INDEX  # noqa
        from models.cmt.transposition import analyze_key, transpose_to_target  # noqa
        from models.cmt.xml_to_cmt_tensors import convert as xml_to_tensors  # noqa
        from utils.utils import pitch_to_midi  # noqa

        with open(self.config["hparams_yaml_path"]) as f:
            hparams = yaml.safe_load(f)
        model_cfg = hparams["model"]
        self.num_bars = int(model_cfg["num_bars"])
        self.frame_per_bar = int(model_cfg["frame_per_bar"])
        num_pitch = int(model_cfg["num_pitch"])
        self.pitch_range = num_pitch - 2

        self.device = torch.device(self.config["device"])
        self.model = ChordConditionedMelodyTransformer(**model_cfg).to(self.device)
        ckpt = torch.load(
            self.config["checkpoint_path"], map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.checkpoint_epoch = ckpt.get("epoch", None)

        self._ChordConditionedMelodyTransformer = ChordConditionedMelodyTransformer
        self._analyze_key = analyze_key
        self._transpose_to_target = transpose_to_target
        self._xml_to_tensors = xml_to_tensors
        self._pitch_to_midi = pitch_to_midi
        self._MELODY_INSTRUMENT_INDEX = MELODY_INSTRUMENT_INDEX

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()

        parsed_stream = m21.converter.parse(req["musicxml_path"])
        key = self._analyze_key(parsed_stream)
        transposed, semitones = self._transpose_to_target(parsed_stream, key)
        tensors = self._xml_to_tensors(
            transposed,
            num_bars=self.num_bars,
            theme_bars=int(req["output_bars"]),
            frame_per_bar=self.frame_per_bar,
            pitch_range=self.pitch_range,
        )

        torch.manual_seed(int(req["seed"]))
        rhythm_t = torch.from_numpy(tensors["rhythm"]).long().unsqueeze(0).to(self.device)
        pitch_t = torch.from_numpy(tensors["pitch"]).long().unsqueeze(0).to(self.device)
        chord_t = torch.from_numpy(tensors["chord"]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model.sampling(rhythm_t, pitch_t, chord_t, topk=int(req["topk"]))
        gen_pitch = result["pitch"][0].cpu().numpy()
        chord_np = chord_t[0].cpu().numpy()

        instruments = self._pitch_to_midi(
            gen_pitch, chord_np[:-1],
            frame_per_bar=self.frame_per_bar,
            basis_note=tensors["base_note"],
        )
        midi_out = pretty_midi.PrettyMIDI()
        midi_out.instruments.append(instruments[self._MELODY_INSTRUMENT_INDEX])
        midi_out_path = Path(req["midi_out_path"])
        midi_out_path.parent.mkdir(parents=True, exist_ok=True)
        midi_out.write(str(midi_out_path))

        return {
            "ok": True,
            "transpose_semitones": -int(semitones),
            "num_bars": self.num_bars,
            "frame_per_bar": self.frame_per_bar,
            "topk": int(req["topk"]),
            "checkpoint_epoch": self.checkpoint_epoch,
        }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--server":
        sys.stderr.write("usage: _subprocess_runner.py --server\n")
        sys.exit(2)

    _add_pipeline_root_to_path()

    config_line = sys.stdin.readline()
    if not config_line:
        sys.stderr.write("no config line on stdin\n")
        sys.exit(2)
    server = _CmtServer(json.loads(config_line))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = server.handle(req)
        except Exception as e:
            resp = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
