"""Comparation-pipeline phase 2 CLI.

Тонкая обёртка над MetricsPhase. Парсит CLI-аргументы, читает YAML,
делегирует в фазу.

CLI: compute_metrics.py --slug <slug>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"

sys.path.insert(0, str(COMP_ROOT))
sys.path.insert(0, str(COMP_ROOT / "metrics"))

from config_loader import load_config  # noqa: E402
from metric_pipelines.registry import all_corpus_pipelines  # noqa: E402
from phases.metrics import (  # noqa: E402
    MetricsPhase, MetricsPhaseError, PerSegmentMetricsRunner,
)


def _per_segment_metrics():
    """Late-import, чтобы pkg-init не делал sys.path-side-effects."""
    from registry import all_metrics  # type: ignore[import-not-found]
    return all_metrics()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compute_metrics.py")
    parser.add_argument("--slug", required=True)
    args = parser.parse_args(argv)

    cfg_path = COMP_ROOT / "configs" / f"{args.slug}.yaml"
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)

    slug_dir = COMP_ROOT / "outputs" / cfg.slug
    if not slug_dir.is_dir():
        print(f"slug output dir not found: {slug_dir}", file=sys.stderr)
        return 2

    try:
        MetricsPhase(
            per_segment_runner=PerSegmentMetricsRunner(metrics=_per_segment_metrics()),
            corpus_pipelines=all_corpus_pipelines(),
        ).run(slug_dir, cfg)
    except MetricsPhaseError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
