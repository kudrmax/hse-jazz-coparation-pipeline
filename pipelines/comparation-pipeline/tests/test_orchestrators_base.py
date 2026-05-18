"""Tests for BaseModelOrchestrator ABC contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from orchestrators.base import BaseModelOrchestrator


def test_base_orchestrator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModelOrchestrator()
