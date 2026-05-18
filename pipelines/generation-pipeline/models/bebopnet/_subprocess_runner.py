"""BebopNet inference subprocess runner — server-mode in models/bebopnet-code/.venv.

Protocol:
  argv:        --server
  first stdin: JSON config (fork_root, model_dir, checkpoint, device)
  loop stdin:  JSON request per line (musicxml_path, seed, output_bars, temperature, midi_out_path)
  loop stdout: JSON response per line (ok=true with top_likelihood, OR ok=false with error+traceback)
  EOF on stdin → graceful exit.

The bidict-shim required to unpickle ``converter_and_duration.pkl`` lives at
the top of ``jazz_rnn.B_next_note_prediction.music_generator``; importing
``MusicGenerator`` from there activates the shim implicitly.

BebopNet's MusicGenerator prints progress messages to stdout (e.g. "assuming
bptt=16"). Both _ensure_loaded and handle redirect stdout → stderr for the
duration of model load and inference, so those lines don't corrupt the JSON
protocol on stdout.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pretty_midi
import torch


class _BebopnetServer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.generator = None
        self.converter = None
        self.device = None
        self._notes_to_stream = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            self._ensure_loaded_inner()
        finally:
            sys.stdout = real_stdout

    def _ensure_loaded_inner(self) -> None:
        sys.path.insert(0, str(Path(self.config["fork_root"])))
        from jazz_rnn.B_next_note_prediction.music_generator import MusicGenerator  # noqa
        from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM  # noqa
        from jazz_rnn.utils.music_utils import notes_to_stream  # noqa

        model_dir = Path(self.config["model_dir"])
        with open(model_dir / "converter_and_duration.pkl", "rb") as f:
            converter = pickle.load(f)
        with open(model_dir / "args.json") as f:
            kwargs = json.load(f)

        self.device = torch.device(self.config["device"])
        self.model = MemTransformerLM(**kwargs)
        ckpt_path = str(model_dir / self.config["checkpoint"])
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
        self.model.converter = converter
        self.model = self.model.to(self.device)
        self.model.eval()

        self.generator = MusicGenerator(
            self.model, converter,
            batch_size=2, beam_width=2, beam_depth=1, beam_search="measure",
            non_stochastic_search=False, top_p=True, temperature=1.0,
            score_model="", threshold=0.0, ensemble=True, song="", no_head=False,
        )
        self.converter = converter
        self._notes_to_stream = notes_to_stream

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()

        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = self._handle_inner(req)
        finally:
            sys.stdout = real_stdout
        return result

    def _handle_inner(self, req: dict[str, Any]) -> dict[str, Any]:
        torch.manual_seed(int(req["seed"]))
        self.generator.temperature = float(req["temperature"])
        self.generator.init_stream(str(req["musicxml_path"]))
        notes, top_likelihood = self.generator.generate_measures(int(req["output_bars"]))

        stream = self._notes_to_stream(
            notes[:, 0, :], self.generator.stream, self.generator.chords,
            self.generator.head_len, False, head_early_start=self.generator.early_start,
        )
        # Strip ChordSymbols (would render as audible chords by m21 midi export).
        for cs in list(stream.recurse().getElementsByClass("ChordSymbol")):
            cs.activeSite.remove(cs)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp_midi = Path(f.name)
        try:
            stream.write("midi", fp=str(tmp_midi))
            midi = pretty_midi.PrettyMIDI(str(tmp_midi))
        finally:
            tmp_midi.unlink(missing_ok=True)

        midi_out_path = Path(req["midi_out_path"])
        midi_out_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(midi_out_path))

        return {"ok": True, "top_likelihood": float(top_likelihood)}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--server":
        sys.stderr.write("usage: _subprocess_runner.py --server\n")
        sys.exit(2)

    config_line = sys.stdin.readline()
    if not config_line:
        sys.stderr.write("no config line on stdin\n")
        sys.exit(2)
    server = _BebopnetServer(json.loads(config_line))

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
