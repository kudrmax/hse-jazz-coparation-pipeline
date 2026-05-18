"""Tests for SelfHealer.sync — sync rules + theme_chunks + chunk-existence."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from corpus import Corpus
from manifest import Manifest
from orchestrators.registry import all_orchestrators
from self_healer import SelfHealer

_AUTUMN_8 = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"


@pytest.fixture
def setup(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    if not _AUTUMN_8.exists():
        pytest.skip("missing source xml")
    return tmp_path, corpus, output, _AUTUMN_8


def _bootstrap(m: Manifest, k: int = 1) -> None:
    m.bootstrap("test", "fp", samples_per_theme=k, output_formats=("midi",))


def test_new_theme_added_with_theme_chunks(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "AllOfMe.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)

    assert "AllOfMe" in m.themes
    assert (output / "themes" / "AllOfMe" / "theme.musicxml").is_file()
    chunks = sorted((output / "themes" / "AllOfMe" / "theme_chunks").glob("chunk_*.musicxml"))
    assert len(chunks) == 1  # 8-bar theme / chunk_bars=8 = 1 chunk


def test_theme_too_short_marks_all_samples_fail(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Tiny.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m, k=2)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=16)

    ts = m.themes["Tiny"]
    for model in ("cmt", "mingus", "bebopnet"):
        for idx in range(2):
            s = ts.models[model].samples[idx]
            assert s.ok is False
            assert "shorter" in (s.error or "").lower()


def test_cmt_raw_chunks_present_promotes_to_ok(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Theme.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    m.add_theme("Theme")
    m.mark_sample("Theme", "cmt", 0, ok=False, error="prev")

    sample_dir = output / "themes" / "Theme" / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.write(str(sample_dir / "raw_chunk_0.mid"))

    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)
    assert m.themes["Theme"].models["cmt"].samples[0].ok is True


def test_mingus_raw_full_present_promotes_to_ok(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Theme.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    m.add_theme("Theme")
    m.mark_sample("Theme", "mingus", 0, ok=False, error="prev")

    sample_dir = output / "themes" / "Theme" / "mingus" / "sample_0"
    sample_dir.mkdir(parents=True)
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.write(str(sample_dir / "raw_full.mid"))

    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)
    assert m.themes["Theme"].models["mingus"].samples[0].ok is True


def test_raw_files_missing_demotes_to_fail(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Theme.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    m.add_theme("Theme")
    m.mark_sample("Theme", "cmt", 0, ok=True, duration_sec=1.0)
    m.mark_sample("Theme", "mingus", 0, ok=True, duration_sec=1.0)

    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)
    assert m.themes["Theme"].models["cmt"].samples[0].ok is False
    assert m.themes["Theme"].models["mingus"].samples[0].ok is False


def test_cmt_partial_raw_chunks_demotes_to_fail(setup):
    """8-bar тема с chunk_bars=4 → 2 чанка ожидаются; только raw_chunk_0 → fail."""
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Theme.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    m.add_theme("Theme")

    sample_dir = output / "themes" / "Theme" / "cmt" / "sample_0"
    sample_dir.mkdir(parents=True)
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.write(str(sample_dir / "raw_chunk_0.mid"))
    # raw_chunk_1.mid отсутствует

    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=4)
    s = m.themes["Theme"].models["cmt"].samples.get(0)
    if s is not None:
        assert s.ok is False


def test_theme_removed_from_corpus_marked(setup):
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "WillBeRemoved.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)
    assert m.themes["WillBeRemoved"].removed_from_corpus is False

    (corpus / "WillBeRemoved.musicxml").unlink()
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)
    assert m.themes["WillBeRemoved"].removed_from_corpus is True


def test_unexpected_slice_error_marks_fail_does_not_crash(setup, monkeypatch):
    """Если slice_score падает с произвольным исключением — pipeline не валится,
    тема помечена fail с конкретной ошибкой, остальные темы обрабатываются."""
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Bad.musicxml")
    shutil.copy2(src, corpus / "Good.musicxml")

    import postprocess
    real_slice = postprocess.slice_score

    def fake_slice(path, chunk_bars):
        if path.parent.name == "Bad":
            raise RuntimeError("simulated m21 export failure")
        return real_slice(path, chunk_bars)

    monkeypatch.setattr("self_healer.slice_score", fake_slice)

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)

    bad = m.themes["Bad"]
    for model in ("cmt", "mingus", "bebopnet"):
        s = bad.models[model].samples[0]
        assert s.ok is False
        assert "slice failed" in (s.error or "").lower()
        assert "RuntimeError" in (s.error or "")

    good = m.themes["Good"]
    chunks = sorted((output / "themes" / "Good" / "theme_chunks").glob("chunk_*.musicxml"))
    assert len(chunks) == 1
    # good is still actionable (sample 0 not yet generated → no ok flag, but no error either)


def test_unexpected_save_error_cleans_up_partial(setup, monkeypatch):
    """Если save_score падает — частичный theme_chunks/ trash'ится, тема fail."""
    _, corpus, output, src = setup
    shutil.copy2(src, corpus / "Bad.musicxml")

    call_count = {"n": 0}
    import postprocess
    real_save = postprocess.save_score

    def fake_save(score, path):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            raise RuntimeError("simulated MusicXMLExportException")
        return real_save(score, path)

    monkeypatch.setattr("self_healer.save_score", fake_save)

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit="all", chunk_bars=8)

    bad = m.themes["Bad"]
    s = bad.models["cmt"].samples[0]
    assert s.ok is False
    assert "save failed" in (s.error or "").lower()
    # theme_chunks/ должна быть trashed (не существовать)
    chunks_dir = output / "themes" / "Bad" / "theme_chunks"
    assert not chunks_dir.exists()


def test_themes_limit_int_excludes_extras(setup):
    _, corpus, output, src = setup
    for n in ["A", "B", "C"]:
        shutil.copy2(src, corpus / f"{n}.musicxml")

    m = Manifest(path=output / "manifest.json")
    _bootstrap(m)
    SelfHealer(all_orchestrators()).sync(m, Corpus(corpus), output, themes_limit=2, chunk_bars=8)
    assert m.themes["A"].removed_from_corpus is False
    assert m.themes["B"].removed_from_corpus is False
    assert "C" not in m.themes
