"""Tests for orchestrators registry."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from orchestrators.bebopnet import BebopnetOrchestrator
from orchestrators.cmt import CmtOrchestrator
from orchestrators.mingus import MingusOrchestrator
from orchestrators.registry import all_orchestrators


def test_all_orchestrators_order():
    orchs = all_orchestrators()
    assert [o.name for o in orchs] == ["cmt", "mingus", "bebopnet"]
    assert isinstance(orchs[0], CmtOrchestrator)
    assert isinstance(orchs[1], MingusOrchestrator)
    assert isinstance(orchs[2], BebopnetOrchestrator)
