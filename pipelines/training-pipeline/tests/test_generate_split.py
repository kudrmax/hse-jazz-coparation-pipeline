"""Tests for split generator. Locks down determinism, CMT compatibility,
and cross-model test-set policy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate_split import (
    EVAL_RATIO,
    EXCLUDED_FROM_TEST,
    TEST_RATIO,
    cmt_base_split,
    generate_split,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MIDI_ROOT = REPO_ROOT / "models" / "CMT-pytorch" / "jazz" / "wjazzd" / "data" / "midi"


def test_generate_split_is_deterministic():
    a = generate_split(MIDI_ROOT)
    b = generate_split(MIDI_ROOT)
    assert a == b


def test_cmt_base_split_partitions_all_solos():
    """CMT-base partition (before excluding) covers all 430 with no duplicates."""
    base = cmt_base_split(MIDI_ROOT)
    all_ids = base["train"] + base["eval"] + base["test"]
    assert len(set(all_ids)) == len(all_ids), "duplicates across buckets"
    assert set(all_ids) == {p.name for p in MIDI_ROOT.iterdir() if p.is_dir()}


def test_cmt_base_split_ratios():
    """CMT base must remain 80/10/10 — locks CMT pkl compatibility."""
    base = cmt_base_split(MIDI_ROOT)
    n_total = sum(len(base[k]) for k in ("train", "eval", "test"))
    assert len(base["eval"]) == int(n_total * EVAL_RATIO)
    assert len(base["test"]) == int(n_total * TEST_RATIO)
    assert len(base["train"]) == n_total - int(n_total * EVAL_RATIO) - int(n_total * TEST_RATIO)


def test_cmt_base_split_matches_seed_logic():
    """Lock down exact reproduction of CMT preprocess.py:178-184."""
    import random
    song_titles = sorted({p.name for p in MIDI_ROOT.iterdir() if p.is_dir()})
    n = len(song_titles)
    num_eval = int(n * 0.1)
    num_test = int(n * 0.1)
    random.seed(0)
    expected_eval = set(random.sample(sorted(set(song_titles)), num_eval))
    expected_test = set(random.sample(sorted(set(song_titles) - expected_eval), num_test))

    base = cmt_base_split(MIDI_ROOT)
    assert set(base["eval"]) == expected_eval
    assert set(base["test"]) == expected_test


def test_generate_split_excludes_unsupported_files_from_test():
    """generate_split removes EXCLUDED_FROM_TEST from test bucket so all
    models test on the same intersection."""
    split = generate_split(MIDI_ROOT)
    for excluded in EXCLUDED_FROM_TEST:
        assert excluded not in split["test"]


def test_generate_split_keeps_excluded_files_out_of_train_and_eval():
    """Excluded files are dropped, NOT moved to train. CMT pkl is locked to
    the original CMT-base partition; moving files would invalidate it."""
    base = cmt_base_split(MIDI_ROOT)
    split = generate_split(MIDI_ROOT)
    # train/eval unchanged from base
    assert split["train"] == base["train"]
    assert split["eval"] == base["eval"]


def test_generate_split_test_size_is_base_minus_excluded():
    """Test = CMT-base test minus EXCLUDED_FROM_TEST that were in base test."""
    base = cmt_base_split(MIDI_ROOT)
    split = generate_split(MIDI_ROOT)
    excluded_in_base_test = set(base["test"]) & set(EXCLUDED_FROM_TEST)
    assert len(split["test"]) == len(base["test"]) - len(excluded_in_base_test)


def test_generate_split_with_empty_excluded_equals_base():
    """When no exclusions, generate_split must equal cmt_base_split exactly."""
    base = cmt_base_split(MIDI_ROOT)
    split = generate_split(MIDI_ROOT, excluded_from_test=[])
    assert split == base


def test_generate_split_raises_on_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="midi_root"):
        generate_split(missing)


def test_generate_split_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="no song subdirectories"):
        generate_split(tmp_path)
