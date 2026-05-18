"""Corpus = source of themes (effendi-fakebook/cleared) + active subset
по themes_limit + helper для count_bars (для resolve auto-bars в orchestrator)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal


class Corpus:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def active_subset(self, themes_limit: int | Literal["all"]) -> list[Path]:
        """sorted(root.rglob('*.musicxml'))[:N]. Детерминированный порядок."""
        all_paths = sorted(self.root.rglob("*.musicxml"))
        if themes_limit == "all":
            return all_paths
        return all_paths[:themes_limit]

    def count_bars(self, theme_path: Path) -> int:
        """Считает Measure'ов через m21. Late-import — m21 тяжёлый."""
        import music21 as m21
        score = m21.converter.parse(str(theme_path))
        part = score.parts[0]
        measures = [m for m in part.getElementsByClass("Measure") if m.number > 0]
        return len(measures)
