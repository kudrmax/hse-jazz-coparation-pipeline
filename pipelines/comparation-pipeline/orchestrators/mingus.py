"""MingusOrchestrator: per-theme full-then-slice (MINGUS specifics)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pretty_midi

from comparation_config import ComparationConfig
from corpus import Corpus
from manifest import Manifest
from orchestrators.base import BaseModelOrchestrator
from postprocess import extract_generated, slice_midi


class MingusOrchestrator(BaseModelOrchestrator):
    name = "mingus"

    def _build_tasks(
        self, cfg: ComparationConfig, manifest: Manifest,
        output_dir: Path, corpus: Corpus,
    ) -> list[dict]:
        SEED_BASE = 42
        tasks: list[dict] = []
        themes_root = output_dir / "themes"
        mb = cfg.mingus
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            ms = ts.models["mingus"]
            resolved_input: int | str = mb.input_bars
            resolved_output: int | str = mb.output_bars
            if mb.input_bars == "auto" or mb.output_bars == "auto":
                n = corpus.count_bars(theme_dir / "theme.musicxml")
                resolved_input = n if mb.input_bars == "auto" else mb.input_bars
                resolved_output = n if mb.output_bars == "auto" else mb.output_bars

            for idx in range(manifest.samples_per_theme):
                sample = ms.samples.get(idx)
                if sample is not None and sample.ok:
                    continue
                sample_dir = theme_dir / "mingus" / f"sample_{idx}"
                if (sample_dir / "raw_full.mid").exists():
                    continue
                task: dict = {
                    "task_id": f"{theme_name}/{idx}",
                    "input": str(theme_dir / "theme.musicxml"),
                    "output_dir": str(sample_dir),
                    "output_stem": "raw_full",
                    "seed": SEED_BASE + idx,
                }
                if mb.input_bars == "auto" or mb.output_bars == "auto":
                    task["input_bars"] = resolved_input
                    task["output_bars"] = resolved_output
                tasks.append(task)
        return tasks

    def _post_process(
        self, manifest: Manifest, output_dir: Path,
        results: dict[str, dict], returncode: int, cfg: ComparationConfig,
    ) -> None:
        themes_root = output_dir / "themes"
        chunk_bars = cfg.segmentation.chunk_bars
        corrupted: set[tuple[str, int]] = set()

        # Pass 1: scan-and-fill
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            model_dir = theme_dir / "mingus"
            if not model_dir.is_dir():
                continue
            for sample_dir in sorted(model_dir.glob("sample_*")):
                if not sample_dir.is_dir():
                    continue
                try:
                    idx = int(sample_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                raw = sample_dir / "raw_full.mid"
                if not raw.exists():
                    continue
                gen_full = sample_dir / "gen_full.mid"
                has_any_chunk = any(sample_dir.glob("gen_chunk_*.mid"))
                if gen_full.exists() and has_any_chunk:
                    continue
                try:
                    pm = pretty_midi.PrettyMIDI(str(raw))
                except Exception as e:
                    print(f"[mingus] corrupted raw {raw}, trashing: {e}", file=sys.stderr)
                    if raw.exists():
                        subprocess.run(["trash", str(raw)], check=True)
                    corrupted.add((theme_name, idx))
                    continue
                try:
                    gen_pm = extract_generated(pm)
                    gen_pm.write(str(gen_full))
                    chunks = slice_midi(gen_pm, chunk_bars)
                    for j, chunk_pm in enumerate(chunks):
                        chunk_pm.write(str(sample_dir / f"gen_chunk_{j}.mid"))
                except Exception as e:
                    print(f"[mingus] post-process failed for {raw}: {e}", file=sys.stderr)

        # Pass 2: update manifest
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            for idx in range(cfg.samples_per_theme):
                sample_dir = theme_dir / "mingus" / f"sample_{idx}"
                if (theme_name, idx) in corrupted:
                    manifest.mark_sample(
                        theme_name, "mingus", idx, ok=False,
                        error="raw_full corrupted, trashed; will regenerate on next run",
                    )
                    continue
                if (sample_dir / "raw_full.mid").exists():
                    manifest.mark_sample(theme_name, "mingus", idx, ok=True)
                else:
                    tid = f"{theme_name}/{idx}"
                    r = results.get(tid)
                    err = (
                        r.get("error") if (r is not None and not r.get("ok"))
                        else "raw_full.mid missing after batch"
                    )
                    if returncode != 0 and tid not in results:
                        err = f"subprocess_died (run.py --batch exit {returncode})"
                    manifest.mark_sample(theme_name, "mingus", idx, ok=False, error=err)

    def _derived_yaml_block(self, cfg: ComparationConfig) -> dict[str, Any]:
        c = cfg.mingus
        return {
            "model": "mingus",
            "output": {
                "formats": list(cfg.output_formats),
                "force_overwrite": True,
            },
            "common": {
                "seed": 1,
                "input_bars": c.input_bars,
                "output_bars": c.output_bars,
                "device": cfg.device,
            },
            "mingus": {
                "fork_root": str(c.fork_root),
                "data_path": str(c.data_path),
                "checkpoint_dir": str(c.checkpoint_dir),
                "epochs": c.epochs,
                "cond_pitch": c.cond_pitch,
                "cond_duration": c.cond_duration,
                "temperature": c.temperature,
            },
        }

    def sample_complete(self, sample_dir: Path, n_chunks: int) -> bool:
        return (sample_dir / "raw_full.mid").exists()
