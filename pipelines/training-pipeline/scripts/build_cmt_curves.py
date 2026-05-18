#!/usr/bin/env python3
"""Build CMT train/val loss curves for all three ablation series.

Run from repo root:
    python pipelines/training-pipeline/scripts/build_cmt_curves.py [SERIES]

SERIES (positional, optional):
    window-length  — 8 / 16 / 32 bars (paper-default остальные параметры)
    dropout        — 16bars dropout 0.2 / 0.3 / 0.4
    num-layers     — 16bars num_layers 8 / 4
    all            — все три (default)

Reads (Phase 2 logs only — финальная фаза обучения, из неё model_best):
    models/CMT-pytorch/result/paper/<run>/training_artefacts/idx002/log.txt

Writes (соответственно):
    pipelines/training-pipeline/figures/cmt-train-val-loss.png
    pipelines/training-pipeline/figures/cmt-train-val-loss-dropout.png
    pipelines/training-pipeline/figures/cmt-train-val-loss-num-layers.png

Style для всех серий — единый: две панели на одной фигуре
(полные кривые epoch 1..100 + zoom на плато epoch 20..100).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "pipelines/training-pipeline/figures"

EPOCH_HEADER_RE = re.compile(r"==========(train|valid) (\d+) epoch==========")
VALUE_RE = re.compile(r"nll/(train|eval):\s*([0-9.eE+-]+)")

ZOOM_XLIM = (20, 100)
ZOOM_YLIM = (1.1, 1.65)  # покрывает все три серии включая 32bars (val ~1.59)


@dataclass(frozen=True)
class Run:
    name: str       # имя папки прогона в result/paper/
    label: str      # подпись в легенде
    color: str      # цвет matplotlib


@dataclass(frozen=True)
class Series:
    key: str             # CLI-ключ
    suptitle: str        # заголовок фигуры
    out_filename: str    # имя png в figures/
    runs: tuple[Run, ...]


SERIES: dict[str, Series] = {
    "window-length": Series(
        key="window-length",
        suptitle="CMT — train/val loss, window-length-sweep (Phase 2)",
        out_filename="cmt-train-val-loss.png",
        runs=(
            Run("8bars",  "8 bars",  "tab:blue"),
            Run("16bars", "16 bars", "tab:orange"),
            Run("32bars", "32 bars", "tab:red"),
        ),
    ),
    "dropout": Series(
        key="dropout",
        suptitle="CMT — train/val loss, dropout-sweep (Phase 2)",
        out_filename="cmt-train-val-loss-dropout.png",
        runs=(
            Run("16bars",           "dropout 0.2", "tab:blue"),
            Run("16bars_dropout03", "dropout 0.3", "tab:orange"),
            Run("16bars_dropout04", "dropout 0.4", "tab:red"),
        ),
    ),
    "num-layers": Series(
        key="num-layers",
        suptitle="CMT — train/val loss, num_layers-sweep (Phase 2)",
        out_filename="cmt-train-val-loss-num-layers.png",
        runs=(
            Run("16bars",     "num_layers = 8", "tab:blue"),
            Run("16bars_nl4", "num_layers = 4", "tab:orange"),
        ),
    ),
}


def log_path(run_name: str) -> Path:
    return (
        REPO_ROOT
        / "models/CMT-pytorch/result/paper"
        / f"{run_name}/training_artefacts/idx002/log.txt"
    )


def parse_log(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    train_epochs: list[int] = []
    train_vals: list[float] = []
    val_epochs: list[int] = []
    val_vals: list[float] = []

    pending_train: int | None = None
    pending_val: int | None = None

    with path.open() as f:
        for line in f:
            m = EPOCH_HEADER_RE.search(line)
            if m:
                phase, ep = m.group(1), int(m.group(2))
                if phase == "train":
                    pending_train = ep
                else:
                    pending_val = ep
                continue

            m = VALUE_RE.search(line)
            if m:
                kind, value = m.group(1), float(m.group(2))
                if kind == "train" and pending_train is not None:
                    train_epochs.append(pending_train)
                    train_vals.append(value)
                    pending_train = None
                elif kind == "eval" and pending_val is not None:
                    val_epochs.append(pending_val)
                    val_vals.append(value)
                    pending_val = None

    return train_epochs, train_vals, val_epochs, val_vals


def render_series(series: Series) -> None:
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    print(f"\n=== {series.key} ===")
    print(f"{'run':<22}{'final train':<14}{'final val':<14}"
          f"{'min val (ep)':<22}{'final - min':<12}")

    for run in series.runs:
        log = log_path(run.name)
        if not log.is_file():
            print(f"  !! log not found: {log}")
            continue

        te, tv, ve, vv = parse_log(log)

        for ax in (ax_full, ax_zoom):
            ax.plot(te, tv, linestyle="-",  color=run.color, alpha=0.9,
                    label=f"{run.label} — train")
            ax.plot(ve, vv, linestyle="--", color=run.color, alpha=0.9,
                    label=f"{run.label} — val")

        if vv:
            min_idx = vv.index(min(vv))
            min_ep, min_val = ve[min_idx], vv[min_idx]
            # По одной вертикали на каждый прогон, color-matched.
            # Помечает эпоху минимума val-loss этого прогона — на этом чекпоинте
            # модель ушла на финальную оценку (best-by-val, см. §2.1.1).
            ax_full.axvline(min_ep, color=run.color, linestyle=":", alpha=0.6)
            ax_zoom.axvline(min_ep, color=run.color, linestyle=":", alpha=0.6)

            final_val = vv[-1]
            final_train = tv[-1] if tv else float("nan")
            gap = final_val - min_val
            print(f"{run.name:<22}{final_train:<14.3f}{final_val:<14.3f}"
                  f"{min_ep:>3} (val={min_val:.3f})        {gap:+.3f}")

    # Доп. запись в легенде, объясняющая смысл точечных вертикалей —
    # одна на все прогоны, нейтральный серый, потому что у каждого прогона
    # своя цветная вертикаль на собственной эпохе минимума val.
    extra_handle = Line2D(
        [], [], color="gray", linestyle=":", alpha=0.7,
        label="эпоха min val (на цвет прогона)",
    )

    ax_full.set_title("Полные кривые (epoch 1..100)")
    ax_full.set_xlabel("epoch")
    ax_full.set_ylabel("nll (combined L_r + L_p)")
    handles_full, labels_full = ax_full.get_legend_handles_labels()
    ax_full.legend(handles_full + [extra_handle],
                   labels_full + [extra_handle.get_label()],
                   fontsize=7, loc="upper right")
    ax_full.grid(alpha=0.3)

    ax_zoom.set_title(
        f"Плато (epoch {ZOOM_XLIM[0]}..{ZOOM_XLIM[1]}, "
        f"ylim {ZOOM_YLIM[0]}..{ZOOM_YLIM[1]})"
    )
    ax_zoom.set_xlim(*ZOOM_XLIM)
    ax_zoom.set_ylim(*ZOOM_YLIM)
    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel("nll (combined L_r + L_p)")
    handles_zoom, labels_zoom = ax_zoom.get_legend_handles_labels()
    ax_zoom.legend(handles_zoom + [extra_handle],
                   labels_zoom + [extra_handle.get_label()],
                   fontsize=7, loc="upper right")
    ax_zoom.grid(alpha=0.3)

    fig.suptitle(series.suptitle)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / series.out_filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "all":
        targets = list(SERIES.values())
    elif arg in SERIES:
        targets = [SERIES[arg]]
    else:
        print(f"Unknown series: {arg!r}. Valid: {list(SERIES.keys())} or 'all'.",
              file=sys.stderr)
        sys.exit(1)

    for series in targets:
        render_series(series)


if __name__ == "__main__":
    main()
