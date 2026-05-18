"""Unit-tests для postprocess.py.

slice_score / save_score — портированы из удалённого test_cmt_slicing.py.
slice_midi — golden behavior (3 chunks из 24-bar PrettyMIDI).
extract_generated — точная граница по downbeats[input_bars].
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from postprocess import (
    ThemeTooShortError,
    extract_generated,
    save_score,
    slice_midi,
    slice_score,
)


_AUTUMN = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"


def _require_xml(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing test fixture: {p}")
    return p


def test_slice_score_returns_one_chunk_for_8bar_theme():
    chunks = slice_score(_require_xml(_AUTUMN), chunk_bars=8)
    assert len(chunks) == 1


def test_slice_score_raises_when_theme_too_short():
    with pytest.raises(ThemeTooShortError):
        slice_score(_require_xml(_AUTUMN), chunk_bars=16)


def test_slice_score_drops_remainder():
    """Autumn_Leaves_8bars (8 bars) с chunk_bars=4 → 2 чанка по 4 такта."""
    chunks = slice_score(_require_xml(_AUTUMN), chunk_bars=4)
    assert len(chunks) == 2


def test_save_score_writes_musicxml(tmp_path):
    chunks = slice_score(_require_xml(_AUTUMN), chunk_bars=8)
    out = tmp_path / "chunk_0.musicxml"
    save_score(chunks[0], out)
    assert out.is_file()
    assert out.read_text().lstrip().startswith("<?xml")


def _make_pm_with_n_bars(n_bars: int, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Создаёт PrettyMIDI 4/4 длиной n_bars тактов с одной нотой в каждом такте.

    Хвост последней ноты выходит за границу n_bars*bar_dur на 0.01s,
    чтобы pretty_midi.get_downbeats() вернул n_bars+1 значений (включая
    границу после последнего такта). Это соответствует реальным
    генерационным MIDI, где ноты не заканчиваются ровно на тактовой границе.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    bar_dur = 60.0 / tempo * 4  # 4/4 at given BPM
    ins = pretty_midi.Instrument(program=0)
    for i in range(n_bars):
        end = (i + 1) * bar_dur
        if i == n_bars - 1:
            end += 0.01
        ins.notes.append(pretty_midi.Note(
            velocity=80, pitch=60, start=i * bar_dur, end=end,
        ))
    pm.instruments.append(ins)
    return pm


def test_slice_midi_3_chunks_from_24_bar_pm():
    pm = _make_pm_with_n_bars(24)
    chunks = slice_midi(pm, chunk_bars=8)
    assert len(chunks) == 3
    for c in chunks:
        n_notes = sum(len(ins.notes) for ins in c.instruments)
        assert n_notes == 8, f"each 8-bar chunk should have 8 notes, got {n_notes}"


def test_slice_midi_drops_remainder():
    pm = _make_pm_with_n_bars(20)
    chunks = slice_midi(pm, chunk_bars=8)
    assert len(chunks) == 2  # 20 // 8 = 2; last 4 bars dropped


def test_slice_midi_empty_when_too_short():
    pm = _make_pm_with_n_bars(4)
    chunks = slice_midi(pm, chunk_bars=8)
    assert chunks == []


def test_extract_generated_midpoint_keeps_second_half():
    """16-bar pm → cut at downbeats[len/2], вторая половина остаётся."""
    pm = _make_pm_with_n_bars(16)
    out = extract_generated(pm)
    n_notes = sum(len(ins.notes) for ins in out.instruments)
    # ~half of 16 notes; depends on get_downbeats() behavior
    assert n_notes >= 7 and n_notes <= 9
    for ins in out.instruments:
        for note in ins.notes:
            assert note.start >= -1e-9


def test_extract_generated_returns_empty_when_pm_too_short():
    pm = _make_pm_with_n_bars(0)  # < 2 downbeats
    out = extract_generated(pm)
    assert sum(len(ins.notes) for ins in out.instruments) == 0
