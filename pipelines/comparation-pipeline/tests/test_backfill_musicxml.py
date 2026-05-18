"""Тесты backfill helper'ов."""
from __future__ import annotations

from pathlib import Path

import music21 as m21

import pretty_midi
import yaml
from backfill_musicxml import backfill_one_chunk, extract_chord_symbols, main


def test_extract_chord_symbols_returns_global_offsets(tmp_path):
    """Chord symbols on consecutive bars should have offsets 0, 4, 8 quarters."""
    score = m21.stream.Score()
    part = m21.stream.Part()
    for bar_idx in range(3):
        m = m21.stream.Measure(number=bar_idx + 1)
        m.append(m21.harmony.ChordSymbol("Cmaj7"))
        m.append(m21.note.Rest(quarterLength=4.0))
        part.append(m)
    score.append(part)
    xml_path = tmp_path / "chunk_0.musicxml"
    score.write("musicxml", fp=str(xml_path))

    chord_syms = extract_chord_symbols(xml_path)
    assert len(chord_syms) == 3
    offsets = [c[0] for c in chord_syms]
    assert offsets == [0.0, 4.0, 8.0]
    for _, cs in chord_syms:
        assert isinstance(cs, m21.harmony.ChordSymbol)


def test_extract_chord_symbols_no_symbols(tmp_path):
    score = m21.stream.Score()
    part = m21.stream.Part()
    m = m21.stream.Measure(number=1)
    m.append(m21.note.Rest(quarterLength=4.0))
    part.append(m)
    score.append(part)
    xml_path = tmp_path / "chunk_0.musicxml"
    score.write("musicxml", fp=str(xml_path))
    assert extract_chord_symbols(xml_path) == []


def _make_theme_chunk_xml(path: Path, chord_symbols: list[tuple[int, str]]) -> None:
    score = m21.stream.Score()
    part = m21.stream.Part()
    for bar_idx in range(8):
        m = m21.stream.Measure(number=bar_idx + 1)
        for offset_in_bar, fig in chord_symbols:
            if offset_in_bar // 4 == bar_idx:
                cs = m21.harmony.ChordSymbol(fig)
                m.insert(offset_in_bar % 4, cs)
        m.append(m21.note.Rest(quarterLength=4.0))
        part.append(m)
    score.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))


def _make_gen_chunk_mid(path: Path) -> None:
    """Одна нота на каждом такте, 8 баров."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    inst = pretty_midi.Instrument(program=0)
    sec_per_bar = 2.0  # 120bpm × 4/4 → 2 sec/bar
    for bar_idx in range(8):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=60, start=bar_idx * sec_per_bar,
            end=bar_idx * sec_per_bar + 1.0,
        ))
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def test_backfill_one_chunk_writes_both_xmls(tmp_path):
    sample_dir = tmp_path / "themes" / "T1" / "cmt" / "sample_0"
    theme_chunks_dir = tmp_path / "themes" / "T1" / "theme_chunks"
    mid_path = sample_dir / "gen_chunk_0.mid"
    theme_chunk = theme_chunks_dir / "chunk_0.musicxml"
    _make_gen_chunk_mid(mid_path)
    _make_theme_chunk_xml(theme_chunk, [(0, "Cmaj7"), (16, "G7")])

    result = backfill_one_chunk(mid_path, theme_chunk, chunk_bars=8)

    assert result == ("written", "written")
    assert (sample_dir / "gen_chunk_0.musicxml").exists()
    assert (sample_dir / "gen_chunk_0_with_chords.musicxml").exists()
    # both must parse cleanly
    melody = m21.converter.parse(str(sample_dir / "gen_chunk_0.musicxml"))
    with_chords = m21.converter.parse(str(sample_dir / "gen_chunk_0_with_chords.musicxml"))
    assert len(melody.recurse().notes) >= 1
    cs_list = list(with_chords.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert len(cs_list) == 2


def test_backfill_one_chunk_idempotent_skip(tmp_path):
    sample_dir = tmp_path / "themes" / "T1" / "cmt" / "sample_0"
    theme_chunks_dir = tmp_path / "themes" / "T1" / "theme_chunks"
    mid_path = sample_dir / "gen_chunk_0.mid"
    theme_chunk = theme_chunks_dir / "chunk_0.musicxml"
    _make_gen_chunk_mid(mid_path)
    _make_theme_chunk_xml(theme_chunk, [(0, "Cmaj7")])

    backfill_one_chunk(mid_path, theme_chunk, chunk_bars=8)
    result = backfill_one_chunk(mid_path, theme_chunk, chunk_bars=8)
    assert result == ("skipped", "skipped")


def test_backfill_one_chunk_missing_theme_chunk_writes_melody_only(tmp_path):
    sample_dir = tmp_path / "themes" / "T1" / "cmt" / "sample_0"
    mid_path = sample_dir / "gen_chunk_0.mid"
    theme_chunk = tmp_path / "missing.musicxml"
    _make_gen_chunk_mid(mid_path)

    result = backfill_one_chunk(mid_path, theme_chunk, chunk_bars=8)

    assert result[0] == "written"
    assert result[1] == "missing_theme_chunk"
    assert (sample_dir / "gen_chunk_0.musicxml").exists()
    assert not (sample_dir / "gen_chunk_0_with_chords.musicxml").exists()


def _bootstrap_minimal_slug(tmp_path: Path) -> Path:
    """Создать минимальный slug на диске: 2 темы × 1 модель × 1 sample × 1 chunk."""
    slug_dir = tmp_path / "outputs" / "test_slug"
    config_snapshot = slug_dir / "config.snapshot.yaml"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot.write_text(yaml.safe_dump({
        "slug": "test_slug",
        "samples_per_theme": 1,
        "device": "cpu",
        "themes_limit": "all",
        "output_formats": ["midi"],
        "segmentation": {"chunk_bars": 8},
        "cmt": {"fork_root": "x", "hparams_yaml_path": "x", "checkpoint_path": "x",
                "topk": 5, "input_bars": 8, "output_bars": 8},
        "mingus": {"fork_root": "x", "data_path": "x", "checkpoint_dir": "x",
                   "epochs": 10, "cond_pitch": "D-C-B-BE-O", "cond_duration": "B-BE-O",
                   "temperature": 1.0, "input_bars": "auto", "output_bars": "auto"},
        "bebopnet": {"fork_root": "x", "model_dir": "x", "checkpoint": "x",
                     "temperature": 1.0, "input_bars": "auto", "output_bars": "auto"},
    }))
    for theme_name in ("T1", "T2"):
        theme_dir = slug_dir / "themes" / theme_name
        _make_theme_chunk_xml(theme_dir / "theme_chunks" / "chunk_0.musicxml", [(0, "Cmaj7")])
        _make_gen_chunk_mid(theme_dir / "cmt" / "sample_0" / "gen_chunk_0.mid")
    return slug_dir


def test_main_walks_full_slug_and_writes_xmls(monkeypatch, tmp_path):
    slug_dir = _bootstrap_minimal_slug(tmp_path)
    monkeypatch.setattr(
        "backfill_musicxml.COMP_ROOT",
        slug_dir.parent.parent,  # tmp_path
    )
    rc = main(["--slug", "test_slug"])
    assert rc == 0
    for theme_name in ("T1", "T2"):
        d = slug_dir / "themes" / theme_name / "cmt" / "sample_0"
        assert (d / "gen_chunk_0.musicxml").exists()
        assert (d / "gen_chunk_0_with_chords.musicxml").exists()
