"""BebopNet duration-vocab dump subprocess.

Loads the bebopnet converter pickle (which requires the bidict shim that
lives in the fork's music_generator.py) and writes its bidict keys as a
[[num, den], ...] JSON list. Pipeline venv reads this back to construct
BebopnetPreprocessor without needing torch / forked music21 in pipeline.
"""
from __future__ import annotations

import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path


def main() -> None:
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    req = json.loads(request_path.read_text())

    fork_root = Path(req["fork_root"])
    sys.path.insert(0, str(fork_root))

    # Import activates the bidict shim required to unpickle the converter.
    import jazz_rnn.B_next_note_prediction.music_generator  # noqa: F401

    pkl_path = Path(req["model_dir"]) / "converter_and_duration.pkl"
    with open(pkl_path, "rb") as f:
        converter = pickle.load(f)

    durations: list[list[int]] = []
    for k in converter.bidict.keys():
        f = Fraction(k)
        durations.append([f.numerator, f.denominator])

    response = {"durations": durations}
    response_path.write_text(json.dumps(response))


if __name__ == "__main__":
    main()
