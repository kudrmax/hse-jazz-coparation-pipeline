"""Registry of all model orchestrators в порядке прогона в GenerationPhase."""
from __future__ import annotations

from orchestrators.base import BaseModelOrchestrator
from orchestrators.bebopnet import BebopnetOrchestrator
from orchestrators.cmt import CmtOrchestrator
from orchestrators.mingus import MingusOrchestrator


def all_orchestrators() -> list[BaseModelOrchestrator]:
    return [CmtOrchestrator(), MingusOrchestrator(), BebopnetOrchestrator()]
