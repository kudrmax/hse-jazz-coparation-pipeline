"""Tests for config_loader.load_config and compute_fingerprint."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pipelines/comparation-pipeline"))

from config_loader import (  # noqa: E402
    ConfigValidationError, compute_fingerprint, load_config,
)


def _valid_yaml_text() -> str:
    """Real-paths YAML using existing model checkpoints in the repo."""
    return f"""
slug: testslug
samples_per_theme: 5
device: cpu
themes_limit: all
output_formats: [midi]
segmentation:
  chunk_bars: 8
cmt:
  fork_root: models/CMT-pytorch
  hparams_yaml_path: models/CMT-pytorch/hparams_jazz_16bars.yaml
  checkpoint_path: models/CMT-pytorch/result/paper/16bars/best_jazz_model_16bars.pth.tar
  topk: 5
  input_bars: 8
  output_bars: 8
mingus:
  fork_root: models/MINGUS
  data_path: models/MINGUS/A_preprocessData/data/DATA.json
  checkpoint_dir: models/MINGUS/B_train/models/paper-optimal
  epochs: 10
  cond_pitch: D-C-B-BE-O
  cond_duration: B-BE-O
  temperature: 1.0
  input_bars: auto
  output_bars: auto
bebopnet:
  fork_root: models/bebopnet-code
  model_dir: models/bebopnet-code/result/paper-default
  checkpoint: model_best.pt
  temperature: 1.0
  input_bars: auto
  output_bars: auto
"""


def _ckpts_available() -> bool:
    return (
        (REPO_ROOT / "models/CMT-pytorch/hparams_jazz_16bars.yaml").exists()
        and (REPO_ROOT / "models/MINGUS/A_preprocessData/data/DATA.json").exists()
        and (REPO_ROOT / "models/bebopnet-code/result/paper-default/model_best.pt").exists()
    )


pytestmark = pytest.mark.skipif(not _ckpts_available(), reason="model checkpoints missing")


def test_loads_valid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "testslug.yaml"
    p.write_text(_valid_yaml_text())
    cfg = load_config(p)
    assert cfg.slug == "testslug"
    assert cfg.samples_per_theme == 5
    assert cfg.themes_limit == "all"
    assert cfg.output_formats == ("midi",)
    assert cfg.segmentation.chunk_bars == 8
    assert cfg.cmt.topk == 5


def test_themes_limit_int(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("themes_limit: all", "themes_limit: 5")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    cfg = load_config(p)
    assert cfg.themes_limit == 5


def test_missing_required_key(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("samples_per_theme: 5\n", "")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="missing keys"):
        load_config(p)


def test_extra_top_level_key(tmp_path: Path) -> None:
    text = _valid_yaml_text() + "\nextra_field: bad\n"
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        load_config(p)


def test_samples_per_theme_zero_fails(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("samples_per_theme: 5", "samples_per_theme: 0")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="samples_per_theme"):
        load_config(p)


def test_output_formats_empty_fails(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("output_formats: [midi]", "output_formats: []")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="output_formats"):
        load_config(p)


def test_output_formats_invalid_fails(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("output_formats: [midi]", "output_formats: [mp3]")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="invalid format"):
        load_config(p)


def test_slug_mismatches_filename_fails(tmp_path: Path) -> None:
    p = tmp_path / "wrongname.yaml"
    p.write_text(_valid_yaml_text())  # slug: testslug, but file = wrongname
    with pytest.raises(ConfigValidationError, match="slug"):
        load_config(p)


def test_invalid_device_fails(tmp_path: Path) -> None:
    text = _valid_yaml_text().replace("device: cpu", "device: gpu")
    p = tmp_path / "testslug.yaml"
    p.write_text(text)
    with pytest.raises(ConfigValidationError, match="device"):
        load_config(p)


def test_fingerprint_stable(tmp_path: Path) -> None:
    """Different key order, same content → same fingerprint."""
    p1 = tmp_path / "v1.yaml"
    p1.write_text(_valid_yaml_text())
    p2 = tmp_path / "v2.yaml"
    # перетасуем top-level keys через парсинг + dump в обратном порядке
    import yaml as _yaml
    raw = _yaml.safe_load(p1.read_text())
    p2.write_text(_yaml.safe_dump(raw, sort_keys=False))  # порядок отличается
    assert compute_fingerprint(p1) == compute_fingerprint(p2)
