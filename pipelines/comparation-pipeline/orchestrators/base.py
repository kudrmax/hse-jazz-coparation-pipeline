"""Base orchestrator для phase 1 (per-model batch generation).

Template Method: `run_batch()` фиксирует sequence; наследники реализуют hooks.
"""
from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from comparation_config import ComparationConfig
    from corpus import Corpus
    from manifest import Manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_RUN_PY = REPO_ROOT / "pipelines/generation-pipeline/runners/run.py"
GEN_VENV_PY = REPO_ROOT / "pipelines/generation-pipeline/.venv/bin/python"


class BaseModelOrchestrator(ABC):
    """Полиморфная стратегия per-model для фазы генерации."""

    name: str  # class-level constant в наследнике: "cmt" | "mingus" | "bebopnet"

    # === Template Method ===
    def run_batch(
        self, cfg: "ComparationConfig", manifest: "Manifest",
        output_dir: Path, corpus: "Corpus",
    ) -> None:
        tasks = self._build_tasks(cfg, manifest, output_dir, corpus)
        returncode, results = self._spawn_batch(cfg, tasks, output_dir)
        self._post_process(manifest, output_dir, results, returncode, cfg)
        manifest.recompute_derived_status()
        manifest.save_atomic()

    # === Hooks (наследник обязан реализовать) ===
    @abstractmethod
    def _build_tasks(
        self, cfg: "ComparationConfig", manifest: "Manifest",
        output_dir: Path, corpus: "Corpus",
    ) -> list[dict]: ...

    @abstractmethod
    def _post_process(
        self, manifest: "Manifest", output_dir: Path,
        results: dict[str, dict], returncode: int, cfg: "ComparationConfig",
    ) -> None: ...

    @abstractmethod
    def _derived_yaml_block(self, cfg: "ComparationConfig") -> dict[str, Any]:
        """Возвращает полный YAML-dict для run.py: {model, output, common, <model>}."""

    @abstractmethod
    def sample_complete(self, sample_dir: Path, n_chunks: int) -> bool:
        """Используется SelfHealer'ом: все ли raw-артефакты семпла на диске?"""

    # === Общая логика (для всех наследников) ===
    def _spawn_batch(
        self, cfg: "ComparationConfig", tasks: list[dict], output_dir: Path,
    ) -> tuple[int, dict[str, dict]]:
        """Spawn run.py --batch. Возвращает (returncode, results_by_task_id)."""
        if not tasks:
            return 0, {}

        derived_yaml = output_dir / "_derived" / f"{self.name}.yaml"
        derived_yaml.parent.mkdir(parents=True, exist_ok=True)
        derived_yaml.write_text(
            yaml.safe_dump(self._derived_yaml_block(cfg),
                           default_flow_style=False, sort_keys=False)
        )

        tasks_path = output_dir / "_derived" / f"{self.name}_tasks.jsonl"
        tasks_path.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")

        results_path = output_dir / "_derived" / f"{self.name}_results.jsonl"
        if results_path.exists():
            results_path.unlink()

        print(f"[{self.name}] running {len(tasks)} tasks ...")
        cmd = [
            str(GEN_VENV_PY), str(GEN_RUN_PY),
            str(derived_yaml),
            "--batch", str(tasks_path),
            "--output", str(results_path),
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT)

        results: dict[str, dict] = {}
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results[r["task_id"]] = r
        return proc.returncode, results
