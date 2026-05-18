"""Тест FinalTableStep — end-to-end на full inputs в tmp."""
from __future__ import annotations

import csv

from final_table.step import FinalTableStep
from final_table.tests.test_builder import _setup_full_inputs


def test_step_writes_both_files(tmp_path):
    slug_dir = tmp_path / "slug"
    _setup_full_inputs(slug_dir / "_metrics")

    FinalTableStep(slug_dir).run()

    assert (slug_dir / "_metrics" / "final.csv").exists()
    assert (slug_dir / "_metrics" / "final_human.csv").exists()


def test_step_final_csv_has_72_rows(tmp_path):
    slug_dir = tmp_path / "slug"
    _setup_full_inputs(slug_dir / "_metrics")
    FinalTableStep(slug_dir).run()

    with (slug_dir / "_metrics" / "final.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 72


def test_step_final_human_csv_has_24_rows(tmp_path):
    slug_dir = tmp_path / "slug"
    _setup_full_inputs(slug_dir / "_metrics")
    FinalTableStep(slug_dir).run()

    with (slug_dir / "_metrics" / "final_human.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
