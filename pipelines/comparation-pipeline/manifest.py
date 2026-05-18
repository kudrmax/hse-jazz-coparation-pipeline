"""Single-JSON manifest for comparation-pipeline state.

Atomic save через tmp+rename (POSIX guarantee). Один writer-process.
Все «derived» поля (theme.status, theme.excluded_due_to, model.status)
пересчитываются через recompute_derived_status() — не хранятся независимо.

См. spec §7.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from model_names import MODEL_NAMES as _MODELS

_Status = Literal["ok", "fail"]


@dataclass
class SampleStatus:
    ok: bool
    duration_sec: float | None = None
    error: str | None = None
    ts: str = ""

    def to_dict(self) -> dict:
        d: dict = {"ok": self.ok}
        if self.duration_sec is not None:
            d["duration_sec"] = self.duration_sec
        if self.error is not None:
            d["error"] = self.error
        if self.ts:
            d["ts"] = self.ts
        return d

    @staticmethod
    def from_dict(d: dict) -> "SampleStatus":
        return SampleStatus(
            ok=bool(d.get("ok")),
            duration_sec=d.get("duration_sec"),
            error=d.get("error"),
            ts=d.get("ts", ""),
        )


@dataclass
class ModelStatus:
    status: _Status = "fail"
    samples: dict[int, SampleStatus] = field(default_factory=dict)


@dataclass
class ThemeStatus:
    status: _Status = "fail"
    removed_from_corpus: bool = False
    excluded_due_to: str | None = None
    models: dict[str, ModelStatus] = field(default_factory=dict)


@dataclass
class Manifest:
    path: Path
    config_slug: str = ""
    config_fingerprint: str = ""
    samples_per_theme: int = 0
    output_formats: tuple[str, ...] = ()
    started_at: str = ""
    last_updated_at: str = ""
    themes: dict[str, ThemeStatus] = field(default_factory=dict)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def active_themes(self) -> list[str]:
        """Темы со status=ok и not removed_from_corpus.

        Канонический фильтр для metric-consumer'ов (см. инвариант C1).
        """
        return [
            name for name, t in self.themes.items()
            if t.status == "ok" and not t.removed_from_corpus
        ]

    def bootstrap(
        self,
        config_slug: str,
        config_fingerprint: str,
        samples_per_theme: int,
        output_formats: tuple[str, ...],
    ) -> None:
        self.config_slug = config_slug
        self.config_fingerprint = config_fingerprint
        self.samples_per_theme = samples_per_theme
        self.output_formats = tuple(output_formats)
        if not self.started_at:
            self.started_at = self._now()

    def add_theme(self, theme: str) -> None:
        if theme in self.themes:
            return
        self.themes[theme] = ThemeStatus(
            models={m: ModelStatus(samples={}) for m in _MODELS}
        )

    def mark_sample(
        self,
        theme: str,
        model: str,
        sample_idx: int,
        *,
        ok: bool,
        duration_sec: float | None = None,
        error: str | None = None,
    ) -> None:
        self.add_theme(theme)
        ms = self.themes[theme].models.setdefault(model, ModelStatus(samples={}))
        ms.samples[sample_idx] = SampleStatus(
            ok=ok, duration_sec=duration_sec, error=error, ts=self._now(),
        )

    def recompute_derived_status(self) -> None:
        for theme_name, ts in self.themes.items():
            # model status — все K семплов должны быть ok (B1)
            for model_name in _MODELS:
                ms = ts.models.setdefault(model_name, ModelStatus(samples={}))
                if (
                    len(ms.samples) >= self.samples_per_theme
                    and all(s.ok for s in ms.samples.values())
                ):
                    ms.status = "ok"
                else:
                    ms.status = "fail"
            # theme status — все 3 модели ok И не removed_from_corpus (C1)
            if ts.removed_from_corpus:
                ts.status = "fail"
                ts.excluded_due_to = "removed_from_corpus"
            elif all(ts.models[m].status == "ok" for m in _MODELS):
                ts.status = "ok"
                ts.excluded_due_to = None
            else:
                ts.status = "fail"
                # первая модель которая упала
                for m in _MODELS:
                    if ts.models[m].status == "fail":
                        ts.excluded_due_to = m
                        break

    def save_atomic(self) -> None:
        self.last_updated_at = self._now()
        data = {
            "config_slug": self.config_slug,
            "config_fingerprint": self.config_fingerprint,
            "samples_per_theme": self.samples_per_theme,
            "output_formats": list(self.output_formats),
            "started_at": self.started_at,
            "last_updated_at": self.last_updated_at,
            "themes": {
                name: {
                    "status": ts.status,
                    "removed_from_corpus": ts.removed_from_corpus,
                    "excluded_due_to": ts.excluded_due_to,
                    "models": {
                        m: {
                            "status": ms.status,
                            "samples": {
                                str(idx): s.to_dict() for idx, s in ms.samples.items()
                            },
                        }
                        for m, ms in ts.models.items()
                    },
                }
                for name, ts in self.themes.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    @staticmethod
    def load(path: Path) -> "Manifest":
        if not path.exists():
            return Manifest(path=path)
        raw = json.loads(path.read_text())
        m = Manifest(
            path=path,
            config_slug=raw.get("config_slug", ""),
            config_fingerprint=raw.get("config_fingerprint", ""),
            samples_per_theme=raw.get("samples_per_theme", 0),
            output_formats=tuple(raw.get("output_formats", [])),
            started_at=raw.get("started_at", ""),
            last_updated_at=raw.get("last_updated_at", ""),
        )
        for name, td in raw.get("themes", {}).items():
            ts = ThemeStatus(
                status=td.get("status", "fail"),
                removed_from_corpus=bool(td.get("removed_from_corpus", False)),
                excluded_due_to=td.get("excluded_due_to"),
            )
            for m_name in _MODELS:
                md = td.get("models", {}).get(m_name, {})
                ms = ModelStatus(status=md.get("status", "fail"))
                for idx_str, sd in md.get("samples", {}).items():
                    ms.samples[int(idx_str)] = SampleStatus.from_dict(sd)
                ts.models[m_name] = ms
            m.themes[name] = ts
        return m

    def failed_pairs(self) -> list[tuple[str, str, int]]:
        """Return list of (theme, model, sample_idx) для всех failed sample'ов."""
        out: list[tuple[str, str, int]] = []
        for theme_name, ts in self.themes.items():
            if ts.removed_from_corpus:
                continue
            for m_name, ms in ts.models.items():
                for idx, s in ms.samples.items():
                    if not s.ok:
                        out.append((theme_name, m_name, idx))
        return out
