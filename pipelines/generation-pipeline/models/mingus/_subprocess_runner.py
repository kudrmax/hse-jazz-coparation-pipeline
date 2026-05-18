"""MINGUS inference subprocess runner — server-mode in models/MINGUS/.venv.

Protocol:
  argv:        --server
  first stdin: JSON config (fork_root, data_path, checkpoint_dir, epochs,
               cond_pitch, cond_duration, device)
  loop stdin:  JSON request per line (musicxml_path, seed, input_bars,
               output_bars, temperature, midi_out_path)
  loop stdout: JSON response per line (ok=true with tempo+title, OR ok=false with error+traceback)
  EOF on stdin → graceful exit.

m21 inside MINGUS venv is 6.7.1 (the fork requires that version).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


class _MingusServer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.modelPitch = None
        self.modelDuration = None
        self.device = None
        # cached imports / vocabs (filled in _ensure_loaded)
        self._xmlToStructuredSong = None
        self._generateOverStandard = None
        self._structuredSongsToPM = None
        self._dbToMusic21 = None
        self._dbToMidiChords = None
        self._dbToChordComposition = None
        self._dbChords = None
        self._vocabPitch = None
        self._vocabDuration = None
        self._pitch_to_ix = None
        self._duration_to_ix = None
        self._beat_to_ix = None
        self._offset_to_ix = None

    def _ensure_loaded(self) -> None:
        if self.modelPitch is not None:
            return
        # MINGUS prints progress messages to stdout (e.g. "Loading data from the
        # Database..."). Redirect stdout → stderr for the duration of loading so
        # those lines don't corrupt our JSON protocol on stdout.
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            self._ensure_loaded_inner()
        finally:
            sys.stdout = real_stdout

    def _ensure_loaded_inner(self) -> None:
        sys.path.insert(0, str(Path(self.config["fork_root"])))
        from B_train.loadDB import MusicDB  # noqa
        from B_train.MINGUS_model import TransformerModel  # noqa
        from C_generate.gen_funct import (  # noqa
            generateOverStandard, structuredSongsToPM, xmlToStructuredSong,
        )

        self.device = torch.device(self.config["device"])
        music_db = MusicDB(
            self.device, 20, 10, 35, False, True, 3,
            data_path=str(self.config["data_path"]),
        )
        vocabPitch, vocabDuration, vocabBeat, vocabOffset = music_db.getVocabs()
        pitch_to_ix, duration_to_ix, beat_to_ix, offset_to_ix = music_db.getInverseVocabs()
        dbChords, dbToMusic21, dbToChordComposition, dbToMidiChords = music_db.getChordDicts()

        pitch_pad_idx = pitch_to_ix["<pad>"]
        duration_pad_idx = duration_to_ix["<pad>"]
        beat_pad_idx = beat_to_ix["<pad>"]
        offset_pad_idx = offset_to_ix["<pad>"]

        chord_encod_dim = 64
        next_chord_encod_dim = 32
        offset_embed_dim = 32
        emsize = 200
        nhid = 200
        nlayers = 4
        nhead = 4
        dropout = 0.2

        # Pitch model (per generate.py:107-141)
        pitch_embed_dim = 512
        duration_embed_dim = 512
        beat_embed_dim = 64
        bass_embed_dim = 64

        self.modelPitch = TransformerModel(
            len(vocabPitch), pitch_embed_dim,
            len(vocabDuration), duration_embed_dim,
            bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
            len(vocabBeat), beat_embed_dim,
            len(vocabOffset), offset_embed_dim,
            emsize, nhead, nhid, nlayers,
            pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
            self.device, dropout, True, self.config["cond_pitch"],
        ).to(self.device)
        pitch_ckpt = (
            Path(self.config["checkpoint_dir"]) / "pitchModel"
            / f"MINGUS COND {self.config['cond_pitch']} Epochs {self.config['epochs']}.pt"
        )
        self.modelPitch.load_state_dict(torch.load(str(pitch_ckpt), map_location=self.device))

        # Duration model (per generate.py:149-183)
        pitch_embed_dim = 64
        duration_embed_dim = 64
        beat_embed_dim = 32
        bass_embed_dim = 32

        self.modelDuration = TransformerModel(
            len(vocabPitch), pitch_embed_dim,
            len(vocabDuration), duration_embed_dim,
            bass_embed_dim, chord_encod_dim, next_chord_encod_dim,
            len(vocabBeat), beat_embed_dim,
            len(vocabOffset), offset_embed_dim,
            emsize, nhead, nhid, nlayers,
            pitch_pad_idx, duration_pad_idx, beat_pad_idx, offset_pad_idx,
            self.device, dropout, False, self.config["cond_duration"],
        ).to(self.device)
        duration_ckpt = (
            Path(self.config["checkpoint_dir"]) / "durationModel"
            / f"MINGUS COND {self.config['cond_duration']} Epochs {self.config['epochs']}.pt"
        )
        self.modelDuration.load_state_dict(torch.load(str(duration_ckpt), map_location=self.device))

        self._xmlToStructuredSong = xmlToStructuredSong
        self._generateOverStandard = generateOverStandard
        self._structuredSongsToPM = structuredSongsToPM
        self._dbToMusic21 = dbToMusic21
        self._dbToMidiChords = dbToMidiChords
        self._dbToChordComposition = dbToChordComposition
        self._dbChords = dbChords
        self._vocabPitch = vocabPitch
        self._vocabDuration = vocabDuration
        self._pitch_to_ix = pitch_to_ix
        self._duration_to_ix = duration_to_ix
        self._beat_to_ix = beat_to_ix
        self._offset_to_ix = offset_to_ix

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()

        # Redirect stdout → stderr during inference: MINGUS prints progress
        # messages (e.g. "Generating over song ...") to stdout that would
        # corrupt our JSON protocol.
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = self._handle_inner(req)
        finally:
            sys.stdout = real_stdout
        return result

    def _handle_inner(self, req: dict[str, Any]) -> dict[str, Any]:
        torch.manual_seed(int(req["seed"]))
        tune = self._xmlToStructuredSong(
            req["musicxml_path"],
            self._dbToMusic21, self._dbToMidiChords, self._dbToChordComposition, self._dbChords,
        )[0]
        # MINGUS counts theme as first chorus → +1 gives `output_bars` of *generated* improv.
        num_chorus = int(req["output_bars"]) // int(req["input_bars"]) + 1
        structured_song = self._generateOverStandard(
            tune, num_chorus, float(req["temperature"]),
            self.modelPitch, self.modelDuration, self._dbToMidiChords,
            self._pitch_to_ix, self._duration_to_ix, self._beat_to_ix, self._offset_to_ix,
            self._vocabPitch, self._vocabDuration,
            35, self.device, False,
        )
        pm = self._structuredSongsToPM(structured_song, self._dbToMidiChords, isJazz=False)
        pm.instruments = pm.instruments[:1]

        midi_out_path = Path(req["midi_out_path"])
        midi_out_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(midi_out_path))

        return {
            "ok": True,
            "tempo": float(structured_song.get("tempo", 0.0)),
            "title": str(structured_song.get("title", Path(req["musicxml_path"]).stem)),
        }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--server":
        sys.stderr.write("usage: _subprocess_runner.py --server\n")
        sys.exit(2)

    config_line = sys.stdin.readline()
    if not config_line:
        sys.stderr.write("no config line on stdin\n")
        sys.exit(2)
    server = _MingusServer(json.loads(config_line))

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
