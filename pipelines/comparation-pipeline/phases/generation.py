"""GenerationPhase: координатор фазы 1 (per-model batch generation).

Знает последовательность шагов: bootstrap output_dir + snapshot YAML,
manifest load + fingerprint-check + bootstrap, self-heal тем, прогон
оркестраторов через Template Method, запись failures.txt, summary.

Не знает специфики моделей — работает с list[BaseModelOrchestrator].
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from comparation_config import ComparationConfig
from config_loader import compute_fingerprint
from corpus import Corpus
from manifest import Manifest
from orchestrators.base import BaseModelOrchestrator
from self_healer import SelfHealer

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = REPO_ROOT / "datasets/effendi-fakebook-refactor/cleared_music_xml"


class GenerationPhase:
    def __init__(self, orchestrators: list[BaseModelOrchestrator]) -> None:
        self.orchestrators = orchestrators

    def run(
        self, cfg: ComparationConfig, yaml_path: Path, output_dir: Path,
    ) -> Manifest:
        output_dir.mkdir(parents=True, exist_ok=True)

        snap = output_dir / "config.snapshot.yaml"
        if not snap.exists():
            shutil.copy2(yaml_path, snap)

        manifest = Manifest.load(output_dir / "manifest.json")
        fingerprint = compute_fingerprint(yaml_path)
        if manifest.config_fingerprint and manifest.config_fingerprint != fingerprint:
            raise GenerationPhaseError(
                f"config fingerprint mismatch:\n"
                f"  manifest: {manifest.config_fingerprint}\n"
                f"  current:  {fingerprint}\n"
                f"либо --force (стереть всё), либо смени slug"
            )
        if not manifest.config_fingerprint:
            manifest.bootstrap(
                config_slug=cfg.slug,
                config_fingerprint=fingerprint,
                samples_per_theme=cfg.samples_per_theme,
                output_formats=cfg.output_formats,
            )

        corpus = Corpus(root=CORPUS_ROOT)
        SelfHealer(self.orchestrators).sync(
            manifest, corpus, output_dir,
            themes_limit=cfg.themes_limit,
            chunk_bars=cfg.segmentation.chunk_bars,
        )
        manifest.save_atomic()

        for orch in self.orchestrators:
            orch.run_batch(cfg, manifest, output_dir, corpus)

        self._write_failures_txt(manifest, output_dir)
        self._print_summary(manifest, output_dir)
        return manifest

    @staticmethod
    def _write_failures_txt(manifest: Manifest, output_dir: Path) -> None:
        lines: list[str] = []
        for theme, model, idx in manifest.failed_pairs():
            s = manifest.themes[theme].models[model].samples[idx]
            lines.append(f"{theme:30} sample_{idx} {model:8} — {s.error or 'unknown'}")
        txt = "\n".join(lines)
        (output_dir / "_failures.txt").write_text(txt + "\n" if txt else "")

    @staticmethod
    def _print_summary(manifest: Manifest, output_dir: Path) -> None:
        n_ok = sum(1 for ts in manifest.themes.values() if ts.status == "ok")
        n_fail = sum(1 for ts in manifest.themes.values() if ts.status == "fail")
        print(f"\nresult: themes ok={n_ok} fail={n_fail}")
        print(f"failures: {output_dir / '_failures.txt'}")
        print(f"manifest: {output_dir / 'manifest.json'}")


class GenerationPhaseError(Exception):
    """Raised on fingerprint mismatch / другие orchestration issues."""
