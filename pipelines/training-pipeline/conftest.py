"""Make `generate_split` importable in tests despite hyphen in dir name.

The directory `pipelines/training-pipeline/` contains a hyphen, which is not
a valid Python module name. Adding it to sys.path lets tests do
`from generate_split import generate_split` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
