"""YAML config loader + validator for generation-pipeline runners/run.py.

Public API:
    load_run_config(yaml_path) -> RunConfig
    ConfigValidationError

All other functions/dataclasses are implementation detail of this module
but are exported for unit testing.

This module deliberately does NOT import torch, music21, or any model
forks — validation is fail-fast and runs from the pipeline venv.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# ----- repo root resolution -----
# pipelines/generation-pipeline/runners/run_config.py → parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ----- exception -----
class ConfigValidationError(Exception):
    """Raised on any YAML schema or value violation. Carries a structured
    field path for tests to assert on without substring-matching messages."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


# ----- dataclasses -----
_FormatLiteral = Literal["midi", "musicxml", "musicxml_with_chords"]
_ModelLiteral = Literal["cmt", "mingus", "bebopnet"]


@dataclass(frozen=True)
class OutputConfig:
    formats: tuple[_FormatLiteral, ...]
    force_overwrite: bool


@dataclass(frozen=True)
class CommonConfig:
    seed: int
    input_bars: int | Literal["auto"]
    output_bars: int | Literal["auto"]
    device: str


@dataclass(frozen=True)
class CmtConfig:
    fork_root: Path
    hparams_yaml_path: Path
    checkpoint_path: Path
    topk: int


@dataclass(frozen=True)
class MingusConfig:
    fork_root: Path
    data_path: Path
    checkpoint_dir: Path
    epochs: int
    cond_pitch: str
    cond_duration: str
    temperature: float


@dataclass(frozen=True)
class BebopnetConfig:
    fork_root: Path
    model_dir: Path
    checkpoint: str
    temperature: float


ModelParams = CmtConfig | MingusConfig | BebopnetConfig


@dataclass(frozen=True)
class RunConfig:
    model: _ModelLiteral
    output: OutputConfig
    common: CommonConfig
    model_params: ModelParams


# ----- path helper -----
def _resolve_path(raw: str) -> Path:
    """Resolve a YAML path to an absolute Path. Absolute paths pass through;
    relative paths are resolved against the repo root."""
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


# ----- public helper: count_bars_from_xml -----
def count_bars_from_xml(path: Path) -> int:
    """Считает количество тактов в musicxml-файле через music21.

    Затакт (Measure с number == 0) игнорируется. Это публичный helper для
    'auto'-режима bars в run.py.

    Импорт music21 — внутри функции, чтобы fail-fast валидация YAML
    оставалась независимой от тяжёлых импортов.
    """
    import music21 as m21  # late import: heavy dep
    score = m21.converter.parse(str(path))
    part = score.parts[0]
    measures = [m for m in part.getElementsByClass("Measure") if m.number > 0]
    return len(measures)


# ----- public entry -----
def load_run_config(yaml_path: Path) -> RunConfig:
    """Load and validate a YAML run config. See spec §3 for all rules.

    Raises ConfigValidationError on any violation. Does not import torch
    or any model forks — purely structural validation."""
    raw = _parse_yaml(yaml_path)
    model = _validate_top_level(raw)
    output = _validate_output(raw["output"])
    common = _validate_common(raw["common"])
    model_params = _validate_model_specific(raw[model], model)
    return RunConfig(model=model, output=output, common=common, model_params=model_params)


# ----- step 1: parse YAML -----
def _parse_yaml(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        raise ConfigValidationError(
            field_path="<config_path>",
            message=f"YAML file does not exist: {yaml_path}",
        )
    try:
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            field_path="<yaml_parse>",
            message=f"failed to parse YAML: {e}",
        ) from e
    if not isinstance(data, dict):
        raise ConfigValidationError(
            field_path="<root>",
            message=f"YAML root must be a mapping, got {type(data).__name__}",
        )
    return data


# ----- step 2-3: top-level keys + model -----
_VALID_MODELS: frozenset[str] = frozenset({"cmt", "mingus", "bebopnet"})


def _validate_top_level(raw: dict[str, Any]) -> _ModelLiteral:
    """Returns the validated `model` value."""
    if "model" not in raw:
        raise ConfigValidationError("model", "required field is missing")
    model = raw["model"]
    if model not in _VALID_MODELS:
        raise ConfigValidationError(
            "model",
            f"unknown model {model!r}; expected one of {sorted(_VALID_MODELS)}",
        )
    expected_keys = {"model", "output", "common", model}
    actual_keys = set(raw.keys())
    extra = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    if missing:
        raise ConfigValidationError(
            "<top-level>",
            f"missing required keys: {sorted(missing)}",
        )
    if extra:
        raise ConfigValidationError(
            "<top-level>",
            f"unknown keys: {sorted(extra)} (expected exactly {sorted(expected_keys)})",
        )
    return model  # type: ignore[return-value]


# ----- step 4: output -----
_VALID_FORMATS: frozenset[str] = frozenset({"midi", "musicxml", "musicxml_with_chords"})
_OUTPUT_REQUIRED_KEYS: frozenset[str] = frozenset({"formats", "force_overwrite"})


def _validate_output(raw: Any) -> OutputConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("output", f"must be a mapping, got {type(raw).__name__}")
    _check_section_keys("output", raw, _OUTPUT_REQUIRED_KEYS)

    formats_raw = raw["formats"]
    if not isinstance(formats_raw, list):
        raise ConfigValidationError("output.formats", f"must be a list, got {type(formats_raw).__name__}")
    if len(formats_raw) == 0:
        raise ConfigValidationError("output.formats", "must be non-empty")
    seen: set[str] = set()
    for i, fmt in enumerate(formats_raw):
        if fmt not in _VALID_FORMATS:
            raise ConfigValidationError(
                f"output.formats[{i}]",
                f"{fmt!r} is not a valid format; expected one of {sorted(_VALID_FORMATS)}",
            )
        if fmt in seen:
            raise ConfigValidationError(f"output.formats[{i}]", f"duplicate format {fmt!r}")
        seen.add(fmt)

    force = raw["force_overwrite"]
    if not isinstance(force, bool):
        raise ConfigValidationError(
            "output.force_overwrite",
            f"must be a boolean, got {type(force).__name__}",
        )

    return OutputConfig(
        formats=tuple(formats_raw),
        force_overwrite=force,
    )


# ----- step 5: common -----
_COMMON_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"seed", "input_bars", "output_bars", "device"}
)
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?|mps)$")


def _validate_common(raw: Any) -> CommonConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("common", f"must be a mapping, got {type(raw).__name__}")
    _check_section_keys("common", raw, _COMMON_REQUIRED_KEYS)

    seed = _require_int("common.seed", raw["seed"])
    input_bars = _require_positive_int_or_auto("common.input_bars", raw["input_bars"])
    output_bars = _require_positive_int_or_auto("common.output_bars", raw["output_bars"])

    device = raw["device"]
    if not isinstance(device, str):
        raise ConfigValidationError("common.device", f"must be a string, got {type(device).__name__}")
    if not _DEVICE_PATTERN.match(device):
        raise ConfigValidationError(
            "common.device",
            f"{device!r} does not match expected pattern (cpu | cuda | cuda:N | mps)",
        )

    return CommonConfig(
        seed=seed,
        input_bars=input_bars,
        output_bars=output_bars,
        device=device,
    )


# ----- step 6: model-specific dispatcher -----
def _validate_model_specific(raw: Any, model: _ModelLiteral) -> ModelParams:
    if not isinstance(raw, dict):
        raise ConfigValidationError(model, f"must be a mapping, got {type(raw).__name__}")
    if model == "cmt":
        return _validate_cmt(raw)
    if model == "mingus":
        return _validate_mingus(raw)
    if model == "bebopnet":
        return _validate_bebopnet(raw)
    raise AssertionError(f"unreachable: model={model!r}")  # _validate_top_level guarantees set


_CMT_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"fork_root", "hparams_yaml_path", "checkpoint_path", "topk"}
)


def _validate_cmt(raw: dict[str, Any]) -> CmtConfig:
    _check_section_keys("cmt", raw, _CMT_REQUIRED_KEYS)
    fork_root = _require_existing_path("cmt.fork_root", raw["fork_root"])
    hparams = _require_existing_path("cmt.hparams_yaml_path", raw["hparams_yaml_path"])
    ckpt = _require_existing_path("cmt.checkpoint_path", raw["checkpoint_path"])
    topk = _require_positive_int("cmt.topk", raw["topk"])
    return CmtConfig(fork_root=fork_root, hparams_yaml_path=hparams, checkpoint_path=ckpt, topk=topk)


_MINGUS_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"fork_root", "data_path", "checkpoint_dir", "epochs", "cond_pitch", "cond_duration", "temperature"}
)


def _validate_mingus(raw: dict[str, Any]) -> MingusConfig:
    _check_section_keys("mingus", raw, _MINGUS_REQUIRED_KEYS)
    fork_root = _require_existing_path("mingus.fork_root", raw["fork_root"])
    data_path = _require_existing_path("mingus.data_path", raw["data_path"])
    ckpt_dir = _require_existing_path("mingus.checkpoint_dir", raw["checkpoint_dir"])
    epochs = _require_positive_int("mingus.epochs", raw["epochs"])
    cond_pitch = _require_str("mingus.cond_pitch", raw["cond_pitch"])
    cond_duration = _require_str("mingus.cond_duration", raw["cond_duration"])
    temperature = _require_positive_float("mingus.temperature", raw["temperature"])
    return MingusConfig(
        fork_root=fork_root,
        data_path=data_path,
        checkpoint_dir=ckpt_dir,
        epochs=epochs,
        cond_pitch=cond_pitch,
        cond_duration=cond_duration,
        temperature=temperature,
    )


_BEBOPNET_REQUIRED_KEYS: frozenset[str] = frozenset({"fork_root", "model_dir", "checkpoint", "temperature"})


def _validate_bebopnet(raw: dict[str, Any]) -> BebopnetConfig:
    _check_section_keys("bebopnet", raw, _BEBOPNET_REQUIRED_KEYS)
    fork_root = _require_existing_path("bebopnet.fork_root", raw["fork_root"])
    model_dir = _require_existing_path("bebopnet.model_dir", raw["model_dir"])
    checkpoint = _require_str("bebopnet.checkpoint", raw["checkpoint"])
    composite = model_dir / checkpoint
    if not composite.exists():
        raise ConfigValidationError(
            "bebopnet.model_dir/bebopnet.checkpoint",
            f"composite path does not exist: {composite}",
        )
    temperature = _require_positive_float("bebopnet.temperature", raw["temperature"])
    return BebopnetConfig(
        fork_root=fork_root, model_dir=model_dir, checkpoint=checkpoint, temperature=temperature
    )


# ----- output dir + collision (called from run.py after CLI parsing) -----
def check_output_collision(output_dir: Path, output: OutputConfig) -> None:
    """Validate the resolved output_dir against the YAML-derived OutputConfig.

    Called from run.py after the user supplies --output-dir on the CLI.
    Raises ConfigValidationError on:
      - output_dir exists but is not a directory
      - target files already exist and output.force_overwrite is False
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigValidationError(
            "--output-dir",
            f"path exists but is not a directory: {output_dir}",
        )
    if not output_dir.is_dir():
        return  # not yet created; nothing to collide with
    if output.force_overwrite:
        return
    stem = output_dir.name
    targets = _target_filenames(stem, output.formats)
    for fmt, target in targets:
        if (output_dir / target).exists():
            raise ConfigValidationError(
                "--output-dir",
                f"target file already exists: {output_dir / target} "
                f"(format={fmt!r}); set output.force_overwrite: true to replace",
            )


def _target_filenames(stem: str, formats: tuple[_FormatLiteral, ...]) -> list[tuple[str, str]]:
    """Map (stem, formats) → list of (format, filename) pairs in declared order."""
    mapping = {
        "midi": f"{stem}.mid",
        "musicxml": f"{stem}.musicxml",
        "musicxml_with_chords": f"{stem}_with_chords.musicxml",
    }
    return [(f, mapping[f]) for f in formats]


# ----- low-level value helpers -----
def _check_section_keys(section: str, raw: dict[str, Any], required: frozenset[str]) -> None:
    actual = set(raw.keys())
    missing = required - actual
    extra = actual - required
    if missing:
        raise ConfigValidationError(section, f"missing required keys: {sorted(missing)}")
    if extra:
        raise ConfigValidationError(
            section, f"unknown keys: {sorted(extra)} (expected exactly {sorted(required)})"
        )


def _require_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(field, f"must be an int, got {type(value).__name__}")
    return value


def _require_positive_int(field: str, value: Any) -> int:
    n = _require_int(field, value)
    if n < 1:
        raise ConfigValidationError(field, f"must be >= 1, got {n}")
    return n


def _require_positive_int_or_auto(field: str, value: Any) -> int | Literal["auto"]:
    """Принимает либо строку 'auto', либо положительное целое."""
    if isinstance(value, str):
        if value == "auto":
            return "auto"
        raise ConfigValidationError(
            field, f"must be a positive int or 'auto', got {value!r}"
        )
    return _require_positive_int(field, value)


def _require_str(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError(field, f"must be a string, got {type(value).__name__}")
    return value


def _require_positive_float(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(field, f"must be a number, got {type(value).__name__}")
    f = float(value)
    if f <= 0:
        raise ConfigValidationError(field, f"must be > 0, got {f}")
    return f


def _require_existing_path(field: str, value: Any) -> Path:
    if not isinstance(value, str):
        raise ConfigValidationError(field, f"must be a string, got {type(value).__name__}")
    p = _resolve_path(value)
    if not p.exists():
        raise ConfigValidationError(field, f"path does not exist: {p}")
    return p
