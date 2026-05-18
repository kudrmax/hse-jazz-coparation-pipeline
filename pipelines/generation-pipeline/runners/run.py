"""Generation pipeline runner — single-input or batch.

Two CLI modes (mutually exclusive):

  Single-input (one .musicxml → outputs in one directory):
      run.py <config.yaml> --input <musicxml> --output-dir <dir>

  Batch (many .musicxml from a JSONL file → outputs per-task + results.jsonl):
      run.py <config.yaml> --batch <tasks.jsonl> --output <results.jsonl>

Both modes go through the same canonical `run_batch(...)` function — single-input
just builds a one-task list internally. The Generator is constructed once per
invocation; for batch mode the forked-venv subprocess (loaded via
PersistentSubprocessClient) is reused across all tasks, so the model checkpoint
is loaded just once per invocation.

YAML carries the run *configuration* (model, seed, bars, device, model
hyperparams, output formats). The CLI carries *what to process this invocation*
(input path(s) and output destination(s)).

The runner lives in the *pipeline venv* (m21 9.9.1, pretty_midi, pyyaml).
Each Generator<X> internally invokes a forked-venv server-mode subprocess for
torch inference (via models/<model>/_subprocess_runner.py --server).

See docs/superpowers/specs/2026-05-08-run-yaml-entrypoint-design.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
RUNNERS_DIR = PIPELINE_ROOT / "runners"
for p in (str(PIPELINE_ROOT), str(RUNNERS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_config import (  # noqa: E402
    BebopnetConfig,
    CmtConfig,
    CommonConfig,
    ConfigValidationError,
    MingusConfig,
    OutputConfig,
    RunConfig,
    check_output_collision,
    count_bars_from_xml,
    load_run_config,
)

from models.base.generator import BaseGenerator  # noqa: E402
from models.base.io import BaseGeneratorInput, BaseGeneratorOutput  # noqa: E402
from models.bebopnet import GeneratorBebopnet, GeneratorBebopnetInput  # noqa: E402
from models.cmt import GeneratorCmt, GeneratorCmtInput  # noqa: E402
from models.mingus import GeneratorMingus, GeneratorMingusInput  # noqa: E402


def _build(cfg: RunConfig, input_path: Path) -> tuple[BaseGenerator, BaseGeneratorInput]:
    """Construct the matching Generator<X> and Generator<X>Input from cfg
    plus the CLI-supplied input musicxml path."""
    common = cfg.common
    match cfg.model_params:
        case CmtConfig() as p:
            gen: BaseGenerator = GeneratorCmt(
                fork_root=p.fork_root,
                hparams_yaml_path=p.hparams_yaml_path,
                checkpoint_path=p.checkpoint_path,
                device=common.device,
            )
            inp: BaseGeneratorInput = GeneratorCmtInput(
                musicxml_path=input_path,
                seed=common.seed,
                input_bars=common.input_bars,
                output_bars=common.output_bars,
                topk=p.topk,
            )
        case MingusConfig() as p:
            gen = GeneratorMingus(
                fork_root=p.fork_root,
                data_path=p.data_path,
                checkpoint_dir=p.checkpoint_dir,
                epochs=p.epochs,
                cond_pitch=p.cond_pitch,
                cond_duration=p.cond_duration,
                device=common.device,
            )
            inp = GeneratorMingusInput(
                musicxml_path=input_path,
                seed=common.seed,
                input_bars=common.input_bars,
                output_bars=common.output_bars,
                temperature=p.temperature,
            )
        case BebopnetConfig() as p:
            gen = GeneratorBebopnet(
                fork_root=p.fork_root,
                model_dir=p.model_dir,
                checkpoint=p.checkpoint,
                device=common.device,
            )
            inp = GeneratorBebopnetInput(
                musicxml_path=input_path,
                seed=common.seed,
                input_bars=common.input_bars,
                output_bars=common.output_bars,
                temperature=p.temperature,
            )
        case _:
            raise AssertionError(f"unreachable: {type(cfg.model_params).__name__}")
    return gen, inp


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Generation pipeline runner — single-input или batch.",
    )
    parser.add_argument("config", type=Path, help="path to YAML config")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="single-input: путь к .musicxml")
    mode.add_argument("--batch", type=Path, help="batch: путь к tasks.jsonl")

    parser.add_argument("--output-dir", type=Path, dest="output_dir",
                        help="single-input: куда писать (basename = stem)")
    parser.add_argument("--output", type=Path, dest="output",
                        help="batch: путь к results.jsonl")
    return parser.parse_args(argv)


def _read_tasks_jsonl(path: Path) -> list[dict]:
    """Загружает список task-dict'ов из JSONL-файла."""
    if not path.exists():
        raise SystemExit(f"tasks file not found: {path}")
    import json
    tasks: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"tasks.jsonl line {i}: invalid JSON: {e}")
        for key in ("task_id", "input", "output_dir", "output_stem"):
            if key not in t:
                raise SystemExit(f"tasks.jsonl line {i}: missing required key {key!r}")
        tasks.append(t)
    if not tasks:
        raise SystemExit(f"tasks.jsonl is empty: {path}")
    return tasks


def _validate_input_path(path: Path) -> Path:
    """Resolve and check the --input musicxml path."""
    resolved = path.resolve()
    if not resolved.exists():
        raise ConfigValidationError("--input", f"path does not exist: {resolved}")
    if not resolved.is_file():
        raise ConfigValidationError("--input", f"not a regular file: {resolved}")
    if resolved.suffix.lower() != ".musicxml":
        raise ConfigValidationError(
            "--input", f"must end with .musicxml, got {resolved.suffix}"
        )
    return resolved


def _resolve_auto_bars(common: CommonConfig, input_path: Path) -> CommonConfig:
    """Если input_bars/output_bars == 'auto', подменяем на count_bars_from_xml.

    Возвращает новый CommonConfig (dataclass frozen). Если оба значения
    числовые — возвращает входной объект без изменений (быстрый путь).
    """
    if common.input_bars != "auto" and common.output_bars != "auto":
        return common
    n = count_bars_from_xml(input_path)
    return CommonConfig(
        seed=common.seed,
        input_bars=n if common.input_bars == "auto" else common.input_bars,
        output_bars=n if common.output_bars == "auto" else common.output_bars,
        device=common.device,
    )


def run_batch(
    cfg: RunConfig,
    tasks: list[dict],
    results_path: Path | None,
) -> None:
    """Единая функция batch-обработки. Используется и --input/--output-dir
    (single-task list) и --batch/--output (multi-task list).

    Зовёт _build один раз → spawn forked subprocess лениво при первом
    gen.generate → close в finally.

    Args:
        cfg: парсированный YAML config.
        tasks: список dict с ключами task_id, input, output_dir, output_stem,
               + опционально seed (override от cfg.common.seed).
        results_path: куда писать results.jsonl, или None для single-input
                      (тогда summary только в stdout).
    """
    import json
    import time

    if not tasks:
        return  # nothing to do

    # Build один раз — конструктор Generator не грузит checkpoint
    # (spawn forked subprocess откладывается до первого _generate_impl).
    gen, _placeholder_inp = _build(cfg, Path(tasks[0]["input"]))

    results_fh = open(results_path, "w") if results_path else None
    try:
        for task in tasks:
            t0 = time.perf_counter()
            try:
                inp = _build_input_for_task(cfg, task)
                out = gen.generate(inp)
                files = _save_outputs_for_task(out, task, cfg.output)
                line: dict = {
                    "task_id": task["task_id"],
                    "ok": True,
                    "files": [str(p) for p in files],
                    "duration_sec": round(time.perf_counter() - t0, 3),
                }
            except Exception as e:
                line = {
                    "task_id": task["task_id"],
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "duration_sec": round(time.perf_counter() - t0, 3),
                }
            _print_summary_line(line)
            if results_fh is not None:
                results_fh.write(json.dumps(line) + "\n")
                results_fh.flush()
    finally:
        if results_fh is not None:
            results_fh.close()
        gen.close()


def _build_input_for_task(cfg: RunConfig, task: dict) -> BaseGeneratorInput:
    """Build Generator<X>Input для одного task'а. Seed берётся из task,
    остальные параметры — из cfg.common + cfg.model_params.

    input_bars/output_bars могут быть overridden через task-поля (для auto-bars
    resolution из orchestrator'а comparation-pipeline).
    """
    common = cfg.common
    seed = int(task.get("seed", common.seed))
    input_path = Path(task["input"])
    # Per-task override: orchestrator пишет resolved int, если cfg был 'auto'
    input_bars = task.get("input_bars", common.input_bars)
    output_bars = task.get("output_bars", common.output_bars)
    match cfg.model_params:
        case CmtConfig() as p:
            return GeneratorCmtInput(
                musicxml_path=input_path,
                seed=seed,
                input_bars=input_bars,
                output_bars=output_bars,
                topk=p.topk,
            )
        case MingusConfig() as p:
            return GeneratorMingusInput(
                musicxml_path=input_path,
                seed=seed,
                input_bars=input_bars,
                output_bars=output_bars,
                temperature=p.temperature,
            )
        case BebopnetConfig() as p:
            return GeneratorBebopnetInput(
                musicxml_path=input_path,
                seed=seed,
                input_bars=input_bars,
                output_bars=output_bars,
                temperature=p.temperature,
            )
    raise AssertionError(f"unreachable: {type(cfg.model_params).__name__}")


def _save_outputs_for_task(
    out: BaseGeneratorOutput,
    task: dict,
    output_cfg: OutputConfig,
) -> list[Path]:
    """Пишет файлы согласно output_cfg.formats. Имя файла = output_stem.
    Collision-check: если файл есть и force_overwrite=False — raise FileExistsError.
    """
    from collections.abc import Callable

    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = task["output_stem"]
    SaveFn = Callable[[Path], Path]
    targets: dict[str, tuple[Path, SaveFn]] = {
        "midi": (output_dir / f"{stem}.mid", out.save_midi),
        "musicxml": (output_dir / f"{stem}.musicxml", out.save_musicxml),
        "musicxml_with_chords": (
            output_dir / f"{stem}_with_chords.musicxml",
            out.save_musicxml_with_chords,
        ),
    }
    written: list[Path] = []
    try:
        for fmt in output_cfg.formats:
            path, save_fn = targets[fmt]
            if path.exists() and not output_cfg.force_overwrite:
                raise FileExistsError(
                    f"output exists: {path} (set output.force_overwrite: true)"
                )
            save_fn(path)
            written.append(path)
    except Exception:
        for p in written:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return written


def _print_summary_line(line: dict) -> None:
    if line["ok"]:
        print(f"[ok] {line['task_id']} ({line['duration_sec']}s) → {line['files']}")
    else:
        print(f"[fail] {line['task_id']}: {line['error']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        cfg = load_run_config(args.config)
        if args.batch:
            if args.output is None:
                raise SystemExit("--batch требует --output")
            tasks = _read_tasks_jsonl(args.batch)
            results_path = args.output.resolve()
        else:
            if args.output_dir is None:
                raise SystemExit("--input требует --output-dir")
            input_path = _validate_input_path(args.input)
            output_dir = args.output_dir.resolve()
            # auto-bars resolution для single-input mode
            resolved_common = _resolve_auto_bars(cfg.common, input_path)
            cfg = RunConfig(
                model=cfg.model, output=cfg.output,
                common=resolved_common, model_params=cfg.model_params,
            )
            check_output_collision(output_dir, cfg.output)
            tasks = [{
                "task_id": "single",
                "input": str(input_path),
                "output_dir": str(output_dir),
                "output_stem": output_dir.name,
            }]
            results_path = None
    except ConfigValidationError as e:
        raise SystemExit(f"config error: {e}") from e

    run_batch(cfg, tasks, results_path)


if __name__ == "__main__":
    main()
