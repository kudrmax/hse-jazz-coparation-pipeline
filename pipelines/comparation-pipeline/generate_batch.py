"""Comparation-pipeline phase 1 CLI.

Тонкая обёртка над GenerationPhase. Парсит CLI-аргументы, читает YAML,
делегирует в фазу.

CLI: generate_batch.py --slug <slug> [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMP_ROOT = REPO_ROOT / "pipelines/comparation-pipeline"

sys.path.insert(0, str(COMP_ROOT))

from config_loader import load_config  # noqa: E402
from orchestrators.registry import all_orchestrators  # noqa: E402
from phases.generation import GenerationPhase, GenerationPhaseError  # noqa: E402


def _trash(path: Path) -> None:
    if path.exists():
        subprocess.run(["trash", str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_batch.py")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--force", action="store_true",
                        help="trash outputs/<slug>/ и начать с нуля")
    args = parser.parse_args(argv)

    yaml_path = COMP_ROOT / "configs" / f"{args.slug}.yaml"
    if not yaml_path.exists():
        print(f"config not found: {yaml_path}", file=sys.stderr)
        return 2
    cfg = load_config(yaml_path)

    output_dir = COMP_ROOT / "outputs" / cfg.slug
    if args.force:
        _trash(output_dir)

    try:
        GenerationPhase(all_orchestrators()).run(cfg, yaml_path, output_dir)
    except GenerationPhaseError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
