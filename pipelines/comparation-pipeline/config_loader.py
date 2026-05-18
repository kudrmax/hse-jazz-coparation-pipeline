"""Loads + validates comparation-pipeline YAML configs (one file per slug).

See spec §10 (formal schema) and §11 (validation rules).
Не импортирует torch / heavy deps; чисто Python + yaml + pathlib.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from comparation_config import (
    BebopnetModelConfig, CmtModelConfig, ComparationConfig, MingusModelConfig,
    SegmentationConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_FORMATS = frozenset({"midi", "musicxml", "musicxml_with_chords"})
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?|mps)$")
_TOP_LEVEL_REQUIRED = frozenset({
    "slug", "samples_per_theme", "device", "themes_limit",
    "output_formats", "segmentation", "cmt", "mingus", "bebopnet",
})
_CMT_REQUIRED = frozenset({
    "fork_root", "hparams_yaml_path", "checkpoint_path", "topk",
    "input_bars", "output_bars",
})
_MINGUS_REQUIRED = frozenset({
    "fork_root", "data_path", "checkpoint_dir", "epochs",
    "cond_pitch", "cond_duration", "temperature", "input_bars", "output_bars",
})
_BEBOPNET_REQUIRED = frozenset({
    "fork_root", "model_dir", "checkpoint", "temperature",
    "input_bars", "output_bars",
})


class ConfigValidationError(Exception):
    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


def load_config(path: Path) -> ComparationConfig:
    raw = _parse_yaml(path)
    _check_keys("<top-level>", raw, _TOP_LEVEL_REQUIRED)
    slug = _require_str("slug", raw["slug"])
    if slug != path.stem:
        raise ConfigValidationError(
            "slug", f"slug ({slug!r}) must match filename ({path.stem!r})"
        )
    samples_per_theme = _require_positive_int("samples_per_theme", raw["samples_per_theme"])
    device = _require_str("device", raw["device"])
    if not _DEVICE_PATTERN.match(device):
        raise ConfigValidationError("device", f"invalid: {device!r}")

    themes_limit_raw = raw["themes_limit"]
    if themes_limit_raw == "all":
        themes_limit: int | str = "all"
    else:
        themes_limit = _require_positive_int("themes_limit", themes_limit_raw)

    output_formats = _validate_output_formats(raw["output_formats"])
    segmentation = _validate_segmentation(raw["segmentation"])
    cmt = _validate_cmt(raw["cmt"])
    mingus = _validate_mingus(raw["mingus"])
    bebopnet = _validate_bebopnet(raw["bebopnet"])

    return ComparationConfig(
        slug=slug, samples_per_theme=samples_per_theme, device=device,
        themes_limit=themes_limit, output_formats=output_formats,
        segmentation=segmentation, cmt=cmt, mingus=mingus, bebopnet=bebopnet,
    )


def compute_fingerprint(path: Path) -> str:
    """SHA256 от parsed-and-normalized YAML (порядок ключей и whitespace
    не влияют). Для config_fingerprint в manifest."""
    raw = yaml.safe_load(path.read_text())
    canonical = yaml.safe_dump(raw, sort_keys=True, default_flow_style=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _parse_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigValidationError("<config>", f"file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigValidationError("<yaml>", f"parse error: {e}") from e
    if not isinstance(data, dict):
        raise ConfigValidationError("<root>", "must be a mapping")
    return data


def _check_keys(field: str, raw: dict, required: frozenset[str]) -> None:
    actual = set(raw)
    missing = required - actual
    extra = actual - required
    if missing:
        raise ConfigValidationError(field, f"missing keys: {sorted(missing)}")
    if extra:
        raise ConfigValidationError(field, f"unknown keys: {sorted(extra)}")


def _require_str(field: str, v: Any) -> str:
    if not isinstance(v, str):
        raise ConfigValidationError(field, f"must be string, got {type(v).__name__}")
    return v


def _require_int(field: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigValidationError(field, f"must be int, got {type(v).__name__}")
    return v


def _require_positive_int(field: str, v: Any) -> int:
    n = _require_int(field, v)
    if n < 1:
        raise ConfigValidationError(field, f"must be >= 1, got {n}")
    return n


def _require_positive_float(field: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigValidationError(field, f"must be number")
    f = float(v)
    if f <= 0:
        raise ConfigValidationError(field, f"must be > 0, got {f}")
    return f


def _resolve_path(field: str, v: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(v, str):
        raise ConfigValidationError(field, "must be string path")
    p = Path(v)
    p = p if p.is_absolute() else (_REPO_ROOT / p).resolve()
    if must_exist and not p.exists():
        raise ConfigValidationError(field, f"path not found: {p}")
    return p


def _require_int_or_auto(field: str, v: Any) -> int | str:
    if isinstance(v, str):
        if v == "auto":
            return "auto"
        raise ConfigValidationError(field, f"must be int or 'auto', got {v!r}")
    return _require_positive_int(field, v)


def _validate_output_formats(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list) or not v:
        raise ConfigValidationError("output_formats", "must be non-empty list")
    seen: set[str] = set()
    for i, fmt in enumerate(v):
        if fmt not in _VALID_FORMATS:
            raise ConfigValidationError(
                f"output_formats[{i}]", f"invalid format {fmt!r}"
            )
        if fmt in seen:
            raise ConfigValidationError(
                f"output_formats[{i}]", f"duplicate {fmt!r}"
            )
        seen.add(fmt)
    return tuple(v)


def _validate_segmentation(raw: Any) -> SegmentationConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("segmentation", "must be mapping")
    _check_keys("segmentation", raw, frozenset({"chunk_bars"}))
    return SegmentationConfig(
        chunk_bars=_require_positive_int("segmentation.chunk_bars", raw["chunk_bars"]),
    )


def _validate_cmt(raw: Any) -> CmtModelConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("cmt", "must be mapping")
    _check_keys("cmt", raw, _CMT_REQUIRED)
    cfg = CmtModelConfig(
        fork_root=_resolve_path("cmt.fork_root", raw["fork_root"]),
        hparams_yaml_path=_resolve_path("cmt.hparams_yaml_path", raw["hparams_yaml_path"]),
        checkpoint_path=_resolve_path("cmt.checkpoint_path", raw["checkpoint_path"]),
        topk=_require_positive_int("cmt.topk", raw["topk"]),
        input_bars=_require_positive_int("cmt.input_bars", raw["input_bars"]),
        output_bars=_require_positive_int("cmt.output_bars", raw["output_bars"]),
    )
    if cfg.input_bars != cfg.output_bars:
        raise ConfigValidationError(
            "cmt.input_bars/output_bars",
            f"CMT requires input_bars==output_bars; got {cfg.input_bars}/{cfg.output_bars}",
        )
    hp = yaml.safe_load(cfg.hparams_yaml_path.read_text())
    expected = int(hp["model"]["num_bars"]) // 2
    if cfg.output_bars != expected:
        raise ConfigValidationError(
            "cmt.output_bars",
            f"must equal hparams.model.num_bars/2 = {expected}, got {cfg.output_bars}",
        )
    return cfg


def _validate_mingus(raw: Any) -> MingusModelConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("mingus", "must be mapping")
    _check_keys("mingus", raw, _MINGUS_REQUIRED)
    return MingusModelConfig(
        fork_root=_resolve_path("mingus.fork_root", raw["fork_root"]),
        data_path=_resolve_path("mingus.data_path", raw["data_path"]),
        checkpoint_dir=_resolve_path("mingus.checkpoint_dir", raw["checkpoint_dir"]),
        epochs=_require_positive_int("mingus.epochs", raw["epochs"]),
        cond_pitch=_require_str("mingus.cond_pitch", raw["cond_pitch"]),
        cond_duration=_require_str("mingus.cond_duration", raw["cond_duration"]),
        temperature=_require_positive_float("mingus.temperature", raw["temperature"]),
        input_bars=_require_int_or_auto("mingus.input_bars", raw["input_bars"]),
        output_bars=_require_int_or_auto("mingus.output_bars", raw["output_bars"]),
    )


def _validate_bebopnet(raw: Any) -> BebopnetModelConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("bebopnet", "must be mapping")
    _check_keys("bebopnet", raw, _BEBOPNET_REQUIRED)
    cfg = BebopnetModelConfig(
        fork_root=_resolve_path("bebopnet.fork_root", raw["fork_root"]),
        model_dir=_resolve_path("bebopnet.model_dir", raw["model_dir"]),
        checkpoint=_require_str("bebopnet.checkpoint", raw["checkpoint"]),
        temperature=_require_positive_float("bebopnet.temperature", raw["temperature"]),
        input_bars=_require_int_or_auto("bebopnet.input_bars", raw["input_bars"]),
        output_bars=_require_int_or_auto("bebopnet.output_bars", raw["output_bars"]),
    )
    composite = cfg.model_dir / cfg.checkpoint
    if not composite.exists():
        raise ConfigValidationError(
            "bebopnet.checkpoint",
            f"composite path not found: {composite}",
        )
    return cfg
