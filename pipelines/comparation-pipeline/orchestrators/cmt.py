"""CmtOrchestrator: per-chunk generation (CMT specifics)."""
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
from postprocess import extract_generated


class CmtOrchestrator(BaseModelOrchestrator):
    name = "cmt"

    def _build_tasks(
        self, cfg: ComparationConfig, manifest: Manifest,
        output_dir: Path, corpus: Corpus,
    ) -> list[dict]:
        SEED_BASE = 42
        tasks: list[dict] = []
        themes_root = output_dir / "themes"
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            ms = ts.models["cmt"]
            for idx in range(manifest.samples_per_theme):
                sample = ms.samples.get(idx)
                if sample is not None and sample.ok:
                    continue
                sample_dir = theme_dir / "cmt" / f"sample_{idx}"
                for j in range(n_chunks):
                    raw_path = sample_dir / f"raw_chunk_{j}.mid"
                    if raw_path.exists():
                        continue
                    tasks.append({
                        "task_id": f"{theme_name}/{idx}/chunk_{j}",
                        "input": str(theme_dir / "theme_chunks" / f"chunk_{j}.musicxml"),
                        "output_dir": str(sample_dir),
                        "output_stem": f"raw_chunk_{j}",
                        "seed": SEED_BASE + idx,
                    })
        return tasks

    def _post_process(
        self, manifest: Manifest, output_dir: Path,
        results: dict[str, dict], returncode: int, cfg: ComparationConfig,
    ) -> None:
        themes_root = output_dir / "themes"
        corrupted_samples: set[tuple[str, int]] = set()

        # Pass 1: scan-and-fill (idempotent)
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            cmt_dir = theme_dir / "cmt"
            if not cmt_dir.is_dir():
                continue
            for sample_dir in sorted(cmt_dir.glob("sample_*")):
                if not sample_dir.is_dir():
                    continue
                try:
                    idx = int(sample_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                for j in range(n_chunks):
                    raw = sample_dir / f"raw_chunk_{j}.mid"
                    gen = sample_dir / f"gen_chunk_{j}.mid"
                    if not raw.exists() or gen.exists():
                        continue
                    try:
                        pm = pretty_midi.PrettyMIDI(str(raw))
                    except Exception as e:
                        print(f"[cmt] corrupted raw {raw}, trashing: {e}", file=sys.stderr)
                        if raw.exists():
                            subprocess.run(["trash", str(raw)], check=True)
                        corrupted_samples.add((theme_name, idx))
                        continue
                    try:
                        gen_pm = extract_generated(pm)
                        gen_pm.write(str(gen))
                    except Exception as e:
                        print(f"[cmt] extract failed for {raw}: {e}", file=sys.stderr)

        # Pass 2: update manifest
        for theme_name, ts in manifest.themes.items():
            if ts.removed_from_corpus:
                continue
            theme_dir = themes_root / theme_name
            n_chunks = len(sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml")))
            if n_chunks == 0:
                continue
            for idx in range(cfg.samples_per_theme):
                sample_dir = theme_dir / "cmt" / f"sample_{idx}"
                if (theme_name, idx) in corrupted_samples:
                    manifest.mark_sample(
                        theme_name, "cmt", idx, ok=False,
                        error="raw chunk corrupted, trashed; will regenerate on next run",
                    )
                    continue
                has_all_raw = all(
                    (sample_dir / f"raw_chunk_{j}.mid").exists() for j in range(n_chunks)
                )
                if has_all_raw:
                    manifest.mark_sample(theme_name, "cmt", idx, ok=True)
                else:
                    first_err = "missing raw chunk(s) after batch"
                    last_tid = ""
                    for j in range(n_chunks):
                        tid = f"{theme_name}/{idx}/chunk_{j}"
                        last_tid = tid
                        r = results.get(tid)
                        if r is not None and not r.get("ok"):
                            first_err = r.get("error", "unknown error")
                            break
                    if returncode != 0 and last_tid not in results:
                        first_err = f"subprocess_died (run.py --batch exit {returncode})"
                    manifest.mark_sample(theme_name, "cmt", idx, ok=False, error=first_err)

    def _derived_yaml_block(self, cfg: ComparationConfig) -> dict[str, Any]:
        c = cfg.cmt
        return {
            "model": "cmt",
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
            "cmt": {
                "fork_root": str(c.fork_root),
                "hparams_yaml_path": str(c.hparams_yaml_path),
                "checkpoint_path": str(c.checkpoint_path),
                "topk": c.topk,
            },
        }

    def sample_complete(self, sample_dir: Path, n_chunks: int) -> bool:
        if not sample_dir.is_dir():
            return False
        return all(
            (sample_dir / f"raw_chunk_{j}.mid").exists()
            for j in range(n_chunks)
        )
