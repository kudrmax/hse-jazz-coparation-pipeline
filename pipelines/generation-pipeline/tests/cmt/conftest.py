"""sys.path setup + shared test helpers for the CMT wrapper test suite.

The round-trip tests need to build, from one description:
- a music21 stream (consumed by our converter)
- a 2-track MIDI in the format the authors' preprocess expects
…and verify the two pipelines agree on the resulting tensors.

NOTE: helpers (build_test_stream, build_parallel_two_track_midi) live in
helpers.py — not here — to avoid name collision with models/CMT-pytorch/conftest.py
which pytest resolves first when a test does `from conftest import ...`.
Import helpers directly: `from helpers import build_test_stream`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
CMT_FORK = REPO_ROOT / "models" / "CMT-pytorch"
CMT_FORK_TESTS = CMT_FORK / "tests"
TESTS_CMT_DIR = REPO_ROOT / "pipelines" / "generation-pipeline" / "tests" / "cmt"

for p in (TESTS_CMT_DIR, PIPELINE_ROOT, CMT_FORK, CMT_FORK_TESTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Re-export helpers for any code that already imported them from here.
from helpers import build_test_stream, build_parallel_two_track_midi  # noqa: E402, F401
