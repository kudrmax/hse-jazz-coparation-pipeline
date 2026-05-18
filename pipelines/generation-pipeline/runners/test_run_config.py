"""Unit tests for run_config.py.

Run from pipeline venv:
    pipelines/generation-pipeline/.venv/bin/python -m pytest \
        pipelines/generation-pipeline/runners/test_run_config.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `runners/` importable: same convention as run_hardcode.py
RUNNERS_DIR = Path(__file__).resolve().parent
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))


def test_module_imports() -> None:
    """Skeleton smoke — verifies the module can be imported at all."""
    import run_config  # noqa: F401


import yaml
import pytest

import run_config
from run_config import (
    ConfigValidationError,
    RunConfig, OutputConfig, CommonConfig,
    CmtConfig, MingusConfig, BebopnetConfig,
    load_run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ----- helpers -----
def _minimal_cmt_dict() -> dict:
    return {
        "model": "cmt",
        "output": {
            "formats": ["midi"],
            "force_overwrite": False,
        },
        "common": {
            "seed": 1,
            "input_bars": 8,
            "output_bars": 8,
            "device": "cpu",
        },
        "cmt": {
            "fork_root": "models/CMT-pytorch",
            "hparams_yaml_path": "models/CMT-pytorch/hparams_jazz_16bars.yaml",
            "checkpoint_path": "models/CMT-pytorch/result/paper/16bars/best_jazz_model_16bars.pth.tar",
            "topk": 5,
        },
    }


def _minimal_mingus_dict() -> dict:
    return {
        "model": "mingus",
        "output": {
            "formats": ["midi"],
            "force_overwrite": False,
        },
        "common": {
            "seed": 1,
            "input_bars": 32,
            "output_bars": 32,
            "device": "cpu",
        },
        "mingus": {
            "fork_root": "models/MINGUS",
            "data_path": "models/MINGUS/A_preprocessData/data/DATA.json",
            "checkpoint_dir": "models/MINGUS/B_train/models/paper-optimal",
            "epochs": 10,
            "cond_pitch": "D-C-B-BE-O",
            "cond_duration": "B-BE-O",
            "temperature": 1.0,
        },
    }


def _minimal_bebopnet_dict() -> dict:
    return {
        "model": "bebopnet",
        "output": {
            "formats": ["midi"],
            "force_overwrite": False,
        },
        "common": {
            "seed": 1,
            "input_bars": 32,
            "output_bars": 32,
            "device": "cpu",
        },
        "bebopnet": {
            "fork_root": "models/bebopnet-code",
            "model_dir": "models/bebopnet-code/result/paper-default",
            "checkpoint": "model_best.pt",
            "temperature": 1.0,
        },
    }


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


# ----- happy-path tests -----
def test_load_minimal_cmt_config(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path, _minimal_cmt_dict())
    cfg = load_run_config(cfg_path)
    assert isinstance(cfg, RunConfig)
    assert cfg.model == "cmt"
    assert isinstance(cfg.model_params, CmtConfig)
    assert cfg.model_params.topk == 5
    assert cfg.common.seed == 1
    assert cfg.common.input_bars == 8
    assert cfg.output.formats == ("midi",)
    assert cfg.output.force_overwrite is False


def test_load_minimal_mingus_config(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path, _minimal_mingus_dict())
    cfg = load_run_config(cfg_path)
    assert isinstance(cfg.model_params, MingusConfig)
    assert cfg.model_params.epochs == 10
    assert cfg.model_params.cond_pitch == "D-C-B-BE-O"
    assert cfg.model_params.cond_duration == "B-BE-O"


def test_load_minimal_bebopnet_config(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path, _minimal_bebopnet_dict())
    cfg = load_run_config(cfg_path)
    assert isinstance(cfg.model_params, BebopnetConfig)
    assert cfg.model_params.checkpoint == "model_best.pt"
    assert cfg.model_params.temperature == 1.0


# ----- top-level validation -----
def test_invalid_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("model: cmt\n  bad indent: yes\n")  # bad YAML
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(p)
    assert exc.value.field_path == "<yaml_parse>"


def test_yaml_path_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(tmp_path / "nonexistent.yaml")
    assert exc.value.field_path == "<config_path>"


def test_yaml_root_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- model\n- cmt\n")
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(p)
    assert exc.value.field_path == "<root>"


def test_missing_model_key(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    del data["model"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "model"


def test_unknown_model(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["model"] = "gpt4"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "model"
    assert "gpt4" in exc.value.message


def test_missing_top_level_key(tmp_path: Path) -> None:
    """Drop one of {output, common, <model>}, expect error."""
    data = _minimal_cmt_dict()
    del data["common"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "<top-level>"
    assert "common" in exc.value.message


def test_extra_top_level_key(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["extra_thing"] = 42
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "<top-level>"
    assert "extra_thing" in exc.value.message


def test_wrong_model_specific_block(tmp_path: Path) -> None:
    """model: cmt but the cmt: block is named mingus:."""
    data = _minimal_cmt_dict()
    data["mingus"] = data.pop("cmt")
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "<top-level>"


def test_two_model_blocks_present(tmp_path: Path) -> None:
    """Both cmt: and bebopnet: present even though model is cmt → extra key error."""
    data = _minimal_cmt_dict()
    data["bebopnet"] = _minimal_bebopnet_dict()["bebopnet"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "<top-level>"
    assert "bebopnet" in exc.value.message


# ----- output section -----
@pytest.mark.parametrize("missing_field", ["formats", "force_overwrite"])
def test_missing_output_field(tmp_path: Path, missing_field: str) -> None:
    data = _minimal_cmt_dict()
    del data["output"][missing_field]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output"
    assert missing_field in exc.value.message


def test_extra_output_field(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["extra"] = "x"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output"
    assert "extra" in exc.value.message


def test_output_formats_not_list(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["formats"] = "midi"  # str, not list
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.formats"


def test_output_formats_empty(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["formats"] = []
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.formats"


def test_output_formats_invalid_value(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["formats"] = ["midi", "pdf"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.formats[1]"
    assert "pdf" in exc.value.message


def test_output_formats_duplicate(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["formats"] = ["midi", "midi"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.formats[1]"


def test_output_formats_independent_with_chords_alone(tmp_path: Path) -> None:
    """musicxml_with_chords without musicxml is allowed (spec §3 rule 4)."""
    data = _minimal_cmt_dict()
    data["output"]["formats"] = ["musicxml_with_chords"]
    cfg_path = _write_yaml(tmp_path, data)
    cfg = load_run_config(cfg_path)
    assert cfg.output.formats == ("musicxml_with_chords",)


def test_force_overwrite_not_bool(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["output"]["force_overwrite"] = "yes"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.force_overwrite"


# Note: output_dir is now a CLI arg, not a YAML field.
# Collision/dir-shape checks live in test_check_output_collision_* below.


# ----- common section -----
@pytest.mark.parametrize(
    "missing_field",
    ["seed", "input_bars", "output_bars", "device"],
)
def test_missing_common_field(tmp_path: Path, missing_field: str) -> None:
    data = _minimal_cmt_dict()
    del data["common"][missing_field]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "common"
    assert missing_field in exc.value.message


def test_extra_common_field(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["common"]["extra"] = 1
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "common"
    assert "extra" in exc.value.message


def test_seed_not_int(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["common"]["seed"] = "abc"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "common.seed"


def test_seed_bool_rejected(tmp_path: Path) -> None:
    """Python bool is a subclass of int; we reject it explicitly."""
    data = _minimal_cmt_dict()
    data["common"]["seed"] = True
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "common.seed"


@pytest.mark.parametrize("field", ["input_bars", "output_bars"])
def test_bars_must_be_positive(tmp_path: Path, field: str) -> None:
    data = _minimal_cmt_dict()
    data["common"][field] = 0
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == f"common.{field}"


# Note: musicxml input path is now a CLI arg (--input), not a YAML field.
# Its validation lives in test_validate_input_path_* below.


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1", "mps"])
def test_device_accepts_valid(tmp_path: Path, device: str) -> None:
    data = _minimal_cmt_dict()
    data["common"]["device"] = device
    cfg_path = _write_yaml(tmp_path, data)
    cfg = load_run_config(cfg_path)
    assert cfg.common.device == device


@pytest.mark.parametrize("device", ["tpu", "gpu", "cuda:", "CUDA", ""])
def test_device_rejects_invalid(tmp_path: Path, device: str) -> None:
    data = _minimal_cmt_dict()
    data["common"]["device"] = device
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "common.device"


# ----- CMT-specific -----
@pytest.mark.parametrize(
    "missing_field", ["fork_root", "hparams_yaml_path", "checkpoint_path", "topk"]
)
def test_missing_cmt_field(tmp_path: Path, missing_field: str) -> None:
    data = _minimal_cmt_dict()
    del data["cmt"][missing_field]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "cmt"
    assert missing_field in exc.value.message


def test_extra_cmt_field(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["cmt"]["temperature"] = 1.0  # would belong to mingus/bebopnet, not cmt
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "cmt"
    assert "temperature" in exc.value.message


@pytest.mark.parametrize(
    "field", ["fork_root", "hparams_yaml_path", "checkpoint_path"]
)
def test_cmt_path_does_not_exist(tmp_path: Path, field: str) -> None:
    data = _minimal_cmt_dict()
    data["cmt"][field] = "no/such/path"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == f"cmt.{field}"


def test_cmt_topk_must_be_positive(tmp_path: Path) -> None:
    data = _minimal_cmt_dict()
    data["cmt"]["topk"] = 0
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "cmt.topk"


# ----- MINGUS-specific -----
@pytest.mark.parametrize(
    "missing_field",
    [
        "fork_root", "data_path", "checkpoint_dir", "epochs",
        "cond_pitch", "cond_duration", "temperature",
    ],
)
def test_missing_mingus_field(tmp_path: Path, missing_field: str) -> None:
    data = _minimal_mingus_dict()
    del data["mingus"][missing_field]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus"
    assert missing_field in exc.value.message


def test_extra_mingus_field(tmp_path: Path) -> None:
    data = _minimal_mingus_dict()
    data["mingus"]["topk"] = 5  # belongs to cmt, not mingus
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus"
    assert "topk" in exc.value.message


@pytest.mark.parametrize("field", ["fork_root", "data_path", "checkpoint_dir"])
def test_mingus_path_does_not_exist(tmp_path: Path, field: str) -> None:
    data = _minimal_mingus_dict()
    data["mingus"][field] = "no/such/path"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == f"mingus.{field}"


def test_mingus_epochs_must_be_positive(tmp_path: Path) -> None:
    data = _minimal_mingus_dict()
    data["mingus"]["epochs"] = 0
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus.epochs"


def test_mingus_temperature_must_be_positive(tmp_path: Path) -> None:
    data = _minimal_mingus_dict()
    data["mingus"]["temperature"] = 0
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus.temperature"


def test_mingus_cond_must_be_string(tmp_path: Path) -> None:
    data = _minimal_mingus_dict()
    data["mingus"]["cond_pitch"] = 42
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus.cond_pitch"


# ----- BebopNet-specific -----
@pytest.mark.parametrize(
    "missing_field", ["fork_root", "model_dir", "checkpoint", "temperature"]
)
def test_missing_bebopnet_field(tmp_path: Path, missing_field: str) -> None:
    data = _minimal_bebopnet_dict()
    del data["bebopnet"][missing_field]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "bebopnet"
    assert missing_field in exc.value.message


def test_extra_bebopnet_field(tmp_path: Path) -> None:
    data = _minimal_bebopnet_dict()
    data["bebopnet"]["topk"] = 5  # cmt-only
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "bebopnet"


@pytest.mark.parametrize("field", ["fork_root", "model_dir"])
def test_bebopnet_path_does_not_exist(tmp_path: Path, field: str) -> None:
    data = _minimal_bebopnet_dict()
    data["bebopnet"][field] = "no/such/path"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == f"bebopnet.{field}"


def test_bebopnet_checkpoint_inside_model_dir(tmp_path: Path) -> None:
    """The checkpoint filename must exist inside model_dir."""
    data = _minimal_bebopnet_dict()
    data["bebopnet"]["checkpoint"] = "no_such_file.pt"
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "bebopnet.model_dir/bebopnet.checkpoint"
    assert "no_such_file.pt" in exc.value.message


def test_bebopnet_temperature_must_be_positive(tmp_path: Path) -> None:
    data = _minimal_bebopnet_dict()
    data["bebopnet"]["temperature"] = -0.5
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "bebopnet.temperature"


# ----- cross-cutting -----
def test_relative_paths_resolve_from_repo_root(tmp_path: Path) -> None:
    """A relative model-specific path resolves to <REPO_ROOT>/<relative>, not CWD-relative."""
    data = _minimal_cmt_dict()
    data["cmt"]["fork_root"] = "models/CMT-pytorch"
    cfg_path = _write_yaml(tmp_path, data)
    cfg = load_run_config(cfg_path)
    assert isinstance(cfg.model_params, CmtConfig)
    expected = (REPO_ROOT / "models" / "CMT-pytorch").resolve()
    assert cfg.model_params.fork_root == expected


def test_field_path_attribute_set(tmp_path: Path) -> None:
    """Tests should be able to assert on .field_path without parsing the message."""
    data = _minimal_cmt_dict()
    data["output"]["formats"] = ["midi", "pdf"]
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "output.formats[1]"
    # Also verify str(exc) matches the documented format
    assert str(exc.value).startswith("output.formats[1]: ")


def test_cross_model_field_in_block_rejected(tmp_path: Path) -> None:
    """topk (cmt-only) inside the mingus: block at model: mingus → extra-key error."""
    data = _minimal_mingus_dict()
    data["mingus"]["topk"] = 5
    cfg_path = _write_yaml(tmp_path, data)
    with pytest.raises(ConfigValidationError) as exc:
        load_run_config(cfg_path)
    assert exc.value.field_path == "mingus"
    assert "topk" in exc.value.message


# ----- check_output_collision (called from run.py after CLI parsing) -----
def test_collision_dir_is_existing_file(tmp_path: Path) -> None:
    file_pretending_to_be_dir = tmp_path / "imafile.txt"
    file_pretending_to_be_dir.write_text("nope")
    output = OutputConfig(formats=("midi",), force_overwrite=False)
    with pytest.raises(ConfigValidationError) as exc:
        run_config.check_output_collision(file_pretending_to_be_dir, output)
    assert exc.value.field_path == "--output-dir"


def test_collision_dir_does_not_exist_yet(tmp_path: Path) -> None:
    """Dir not created yet → no collision possible, passes."""
    output = OutputConfig(formats=("midi",), force_overwrite=False)
    run_config.check_output_collision(tmp_path / "not_yet", output)  # no raise


def test_collision_target_exists_no_force(tmp_path: Path) -> None:
    out_dir = tmp_path / "run1"
    out_dir.mkdir()
    (out_dir / "run1.mid").write_bytes(b"")
    output = OutputConfig(formats=("midi",), force_overwrite=False)
    with pytest.raises(ConfigValidationError) as exc:
        run_config.check_output_collision(out_dir, output)
    assert exc.value.field_path == "--output-dir"
    assert "run1.mid" in exc.value.message


def test_collision_target_exists_with_force(tmp_path: Path) -> None:
    out_dir = tmp_path / "run1"
    out_dir.mkdir()
    (out_dir / "run1.mid").write_bytes(b"")
    output = OutputConfig(formats=("midi",), force_overwrite=True)
    run_config.check_output_collision(out_dir, output)  # no raise


def test_collision_only_checks_target_formats(tmp_path: Path) -> None:
    """Pre-existing musicxml does NOT block a midi-only run."""
    out_dir = tmp_path / "run1"
    out_dir.mkdir()
    (out_dir / "run1.musicxml").write_text("<score/>")  # exists but not in formats
    output = OutputConfig(formats=("midi",), force_overwrite=False)
    run_config.check_output_collision(out_dir, output)  # no raise


# ----- run.py CLI helpers (input validation + _save_outputs rollback) -----
def _stub_heavy_imports_and_load_run(monkeypatch: pytest.MonkeyPatch) -> object:
    """Make run.py importable from the pipeline venv by stubbing torch-deps.

    All sys.modules mutations go through `monkeypatch.setitem`, so they are
    automatically rolled back at teardown — without that, the stubs would
    overwrite real model classes (e.g. BaseGeneratorInput becomes a bare
    `type(...)` with no __init__) and break unrelated tests later in the run.
    Real generators are exercised in end-to-end smoke from a model venv."""
    monkeypatch.syspath_prepend(
        str((REPO_ROOT / "pipelines" / "generation-pipeline" / "runners").resolve())
    )
    import types

    def _fake_module(name: str, **attrs: object) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    fakes = {
        "models": _fake_module("models"),
        "models.base": _fake_module("models.base"),
        "models.base.generator": _fake_module(
            "models.base.generator",
            BaseGenerator=type("BaseGenerator", (), {}),
        ),
        "models.base.io": _fake_module(
            "models.base.io",
            BaseGeneratorInput=type("BaseGeneratorInput", (), {}),
            BaseGeneratorOutput=type("BaseGeneratorOutput", (), {}),
        ),
        "models.bebopnet": _fake_module(
            "models.bebopnet",
            GeneratorBebopnet=type("GeneratorBebopnet", (), {}),
            GeneratorBebopnetInput=type("GeneratorBebopnetInput", (), {}),
        ),
        "models.cmt": _fake_module(
            "models.cmt",
            GeneratorCmt=type("GeneratorCmt", (), {}),
            GeneratorCmtInput=type("GeneratorCmtInput", (), {}),
        ),
        "models.mingus": _fake_module(
            "models.mingus",
            GeneratorMingus=type("GeneratorMingus", (), {}),
            GeneratorMingusInput=type("GeneratorMingusInput", (), {}),
        ),
    }
    for name, fake in fakes.items():
        monkeypatch.setitem(sys.modules, name, fake)
    # `run` itself must also be removed at teardown — otherwise a stale copy
    # built against fake modules persists for later tests.
    monkeypatch.delitem(sys.modules, "run", raising=False)
    import run
    return run


def test_validate_input_path_does_not_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = _stub_heavy_imports_and_load_run(monkeypatch)
    with pytest.raises(ConfigValidationError) as exc:
        run._validate_input_path(tmp_path / "no_such.musicxml")
    assert exc.value.field_path == "--input"


def test_validate_input_path_wrong_extension(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "fake.txt"
    fake.write_text("not musicxml")
    run = _stub_heavy_imports_and_load_run(monkeypatch)
    with pytest.raises(ConfigValidationError) as exc:
        run._validate_input_path(fake)
    assert exc.value.field_path == "--input"


def test_validate_input_path_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _stub_heavy_imports_and_load_run(monkeypatch)
    real = (REPO_ROOT / "pipelines" / "generation-pipeline" / "inputs" / "musicxml"
            / "Autumn_Leaves_8bars.musicxml")
    resolved = run._validate_input_path(real)
    assert resolved == real.resolve()


def test_save_outputs_rolls_back_partial_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If save_musicxml fails, the just-written .mid is removed; pre-existing
    files are NOT touched."""
    run = _stub_heavy_imports_and_load_run(monkeypatch)

    out_dir = tmp_path / "rollback_test"
    out_dir.mkdir()

    # Pre-existing file we must NOT touch (different name, not a target).
    (out_dir / "preexisting.txt").write_text("keep me")

    output_cfg = run.OutputConfig(
        formats=("midi", "musicxml"),
        force_overwrite=False,
    )

    # Fake output: save_midi succeeds, save_musicxml raises.
    class FakeOut:
        def save_midi(self, path: Path) -> Path:
            Path(path).write_bytes(b"fake midi")
            return Path(path)

        def save_musicxml(self, path: Path) -> Path:
            raise RuntimeError("simulated failure")

        def save_musicxml_with_chords(self, path: Path) -> Path:
            raise AssertionError("should not be called")

    task = {"output_dir": str(out_dir), "output_stem": "rollback_test"}
    with pytest.raises(RuntimeError, match="simulated failure"):
        run._save_outputs_for_task(FakeOut(), task, output_cfg)

    # Just-written .mid removed, pre-existing file kept.
    assert not (out_dir / "rollback_test.mid").exists(), "partial .mid not rolled back"
    assert (out_dir / "preexisting.txt").exists(), "non-target file wrongly removed"


# ----- bars: auto support -----
def test_input_bars_auto_accepted() -> None:
    raw = {"seed": 1, "input_bars": "auto", "output_bars": 8, "device": "cpu"}
    cc = run_config._validate_common(raw)
    assert cc.input_bars == "auto"
    assert cc.output_bars == 8


def test_output_bars_auto_accepted() -> None:
    raw = {"seed": 1, "input_bars": 8, "output_bars": "auto", "device": "cpu"}
    cc = run_config._validate_common(raw)
    assert cc.input_bars == 8
    assert cc.output_bars == "auto"


def test_both_bars_auto_accepted() -> None:
    raw = {"seed": 1, "input_bars": "auto", "output_bars": "auto", "device": "cpu"}
    cc = run_config._validate_common(raw)
    assert cc.input_bars == "auto"
    assert cc.output_bars == "auto"


def test_input_bars_invalid_string_rejected() -> None:
    raw = {"seed": 1, "input_bars": "bla", "output_bars": 8, "device": "cpu"}
    with pytest.raises(ConfigValidationError) as exc:
        run_config._validate_common(raw)
    assert exc.value.field_path == "common.input_bars"


def test_input_bars_zero_still_rejected() -> None:
    raw = {"seed": 1, "input_bars": 0, "output_bars": 8, "device": "cpu"}
    with pytest.raises(ConfigValidationError) as exc:
        run_config._validate_common(raw)
    assert exc.value.field_path == "common.input_bars"


# ----- count_bars_from_xml -----
def test_count_bars_autumn_leaves_full() -> None:
    """Autumn_Leaves.musicxml — полная форма 32 такта."""
    path = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves.musicxml"
    assert run_config.count_bars_from_xml(path) == 32


def test_count_bars_autumn_leaves_8bars() -> None:
    """Autumn_Leaves_8bars.musicxml — 8 тактов."""
    path = REPO_ROOT / "pipelines/generation-pipeline/inputs/musicxml/Autumn_Leaves_8bars.musicxml"
    assert run_config.count_bars_from_xml(path) == 8


def test_count_bars_ignores_pickup() -> None:
    """Затакт (measure number 0) не считается за такт."""
    import music21 as m21
    score = m21.stream.Score()
    part = m21.stream.Part()
    pickup = m21.stream.Measure(number=0)
    pickup.append(m21.note.Note("C4", quarterLength=1.0))
    part.append(pickup)
    for i in range(1, 5):
        m = m21.stream.Measure(number=i)
        m.append(m21.note.Note("C4", quarterLength=4.0))
        part.append(m)
    score.append(part)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
        score.write("musicxml", fp=f.name)
        path = Path(f.name)
    try:
        assert run_config.count_bars_from_xml(path) == 4  # pickup игнорируется
    finally:
        path.unlink()
