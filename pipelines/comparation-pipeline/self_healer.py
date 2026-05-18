"""Sync between manifest, on-disk artifacts, and active corpus subset.

Реализует:
1. Bootstrap themes_root/<theme>/theme.musicxml (копия из corpus).
2. Bootstrap themes_root/<theme>/theme_chunks/chunk_<j>.musicxml — нарезка
   темы на chunk_bars блоки через postprocess.slice_score (вход для CMT
   и эталон для метрик).
3. Sample-existence check (semantically = "subprocess отработал"):
   - CMT: sample ok iff все raw_chunk_<j>.mid (j ∈ [0, n_chunks)) существуют.
   - MINGUS/BebopNet: sample ok iff raw_full.mid существует.
4. removed_from_corpus toggling.

Single entry point: sync().
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from corpus import Corpus
from manifest import Manifest, ModelStatus, SampleStatus
from model_names import MODEL_NAMES as _MODELS
from orchestrators.base import BaseModelOrchestrator
from postprocess import ThemeTooShortError, save_score, slice_score


def _trash(path: Path) -> None:
    if path.exists():
        subprocess.run(["trash", str(path)], check=True)


class SelfHealer:
    def __init__(self, orchestrators: list[BaseModelOrchestrator]) -> None:
        self.orchestrators_by_name = {o.name: o for o in orchestrators}

    def sync(
        self,
        manifest: Manifest,
        corpus: Corpus,
        output_dir: Path,
        themes_limit: int | Literal["all"],
        chunk_bars: int,
    ) -> None:
        active_paths = corpus.active_subset(themes_limit)
        active_names = {p.stem for p in active_paths}
        themes_root = output_dir / "themes"
        themes_root.mkdir(parents=True, exist_ok=True)

        for theme_path in active_paths:
            name = theme_path.stem
            if name not in manifest.themes:
                manifest.add_theme(name)
            theme_dir = themes_root / name
            theme_dir.mkdir(parents=True, exist_ok=True)
            theme_xml = theme_dir / "theme.musicxml"
            if not theme_xml.exists():
                shutil.copy2(theme_path, theme_xml)

            ts = manifest.themes[name]
            if ts.removed_from_corpus:
                ts.removed_from_corpus = False

            n_chunks, theme_error = self._ensure_theme_chunks(theme_xml, theme_dir, chunk_bars)

            if n_chunks == 0:
                # тема непригодна (короче chunk_bars или slice/save упал) —
                # все семплы fail для всех моделей с конкретной ошибкой
                err = theme_error or "theme unprocessable"
                for model in _MODELS:
                    ms = ts.models.setdefault(model, ModelStatus(samples={}))
                    for idx in range(manifest.samples_per_theme):
                        ms.samples[idx] = SampleStatus(ok=False, error=err)
                continue

            for model in _MODELS:
                model_dir = theme_dir / model
                model_dir.mkdir(parents=True, exist_ok=True)
                ms = ts.models.setdefault(model, ModelStatus(samples={}))
                orch = self.orchestrators_by_name[model]
                for idx in range(manifest.samples_per_theme):
                    sample_dir = model_dir / f"sample_{idx}"
                    has_raw = orch.sample_complete(sample_dir, n_chunks)
                    cur = ms.samples.get(idx)
                    if has_raw:
                        if cur is None or not cur.ok:
                            ms.samples[idx] = SampleStatus(ok=True, duration_sec=None)
                    else:
                        if cur is not None and cur.ok:
                            ms.samples[idx] = SampleStatus(
                                ok=False, error="files missing on disk",
                            )

        for name, ts in manifest.themes.items():
            if name not in active_names:
                ts.removed_from_corpus = True

        manifest.recompute_derived_status()

    def _ensure_theme_chunks(
        self, theme_xml: Path, theme_dir: Path, chunk_bars: int,
    ) -> tuple[int, str | None]:
        """Создаёт theme_chunks/chunk_<j>.musicxml если их ещё нет.

        Возвращает (n_chunks, error):
        - (n>0, None) — успешно нарезано, n чанков на диске
        - (0, "theme shorter than chunk_bars") — тема слишком короткая
        - (0, "theme slice failed: ...") — slice_score бросил неожиданное
        - (0, "theme chunks save failed: ...") — save_score бросил неожиданное

        Любые исключения ловятся, частично сохранённый theme_chunks/ → trash.
        """
        chunks_dir = theme_dir / "theme_chunks"
        existing = sorted(chunks_dir.glob("chunk_*.musicxml")) if chunks_dir.is_dir() else []
        if existing:
            return len(existing), None

        theme_name = theme_xml.parent.name
        try:
            chunks = slice_score(theme_xml, chunk_bars)
        except ThemeTooShortError as e:
            print(f"[self_heal] {theme_name}: {e}", file=sys.stderr)
            return 0, "theme shorter than chunk_bars"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[self_heal] {theme_name}: slice_score failed: {msg}", file=sys.stderr)
            return 0, f"theme slice failed: {msg}"

        try:
            chunks_dir.mkdir(parents=True, exist_ok=True)
            for j, chunk in enumerate(chunks):
                save_score(chunk, chunks_dir / f"chunk_{j}.musicxml")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[self_heal] {theme_name}: save_score failed: {msg}", file=sys.stderr)
            if chunks_dir.is_dir():
                try:
                    _trash(chunks_dir)
                except Exception as trash_err:
                    print(
                        f"[self_heal] {theme_name}: trash {chunks_dir} failed: {trash_err}",
                        file=sys.stderr,
                    )
            return 0, f"theme chunks save failed: {msg}"

        return len(chunks), None
