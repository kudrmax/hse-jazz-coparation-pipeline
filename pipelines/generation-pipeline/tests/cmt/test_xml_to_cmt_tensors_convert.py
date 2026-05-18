"""Tests for the convert() entry point in xml_to_cmt_tensors."""
import numpy as np

from models.cmt.xml_to_cmt_tensors import convert
from helpers import build_test_stream


def test_convert_returns_expected_shape_and_keys():
    melody_notes = [(i * 0.5, 0.5, "C5") for i in range(8)]
    chord_changes = [(0.0, "C"), (4.0, "G")]
    stream = build_test_stream(
        melody_notes=melody_notes,
        chord_changes=chord_changes,
        num_bars=4,
    )
    out = convert(stream, num_bars=4, theme_bars=2, frame_per_bar=16)

    assert set(out.keys()) == {"rhythm", "pitch", "chord", "base_note"}
    assert out["rhythm"].shape == (2 * 16,)            # theme half only
    assert out["pitch"].shape == (2 * 16,)             # theme half only
    assert out["chord"].shape == (4 * 16 + 1, 12)       # full window + terminal frame
    assert isinstance(out["base_note"], int)
