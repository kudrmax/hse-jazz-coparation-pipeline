#!/usr/bin/env python3
"""Build BebopNet val loss curve from epochs.csv.

Run from repo root:
    python pipelines/training-pipeline/scripts/build_bebopnet_curves.py

Reads:
    pipelines/training-pipeline/results/bebopnet-wjazzd-500K/epochs.csv

Writes:
    pipelines/training-pipeline/figures/bebopnet-train-val-loss.png

Note: train loss currently NOT logged in epochs.csv (см. metrics.md, статус WAIT:MAYBE).
График строится только по val_loss до решения вопроса о train loss.

Обозначения на графике:
- основная линия — val combined NLL
- вертикальная пунктирная линия — шаг минимума val (best-by-val checkpoint =
  тот, на котором считается final_test_ppl)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = (
    REPO_ROOT / "pipelines/training-pipeline/results/bebopnet-wjazzd-500K/epochs.csv"
)
OUT_PATH = REPO_ROOT / "pipelines/training-pipeline/figures/bebopnet-train-val-loss.png"


def read_csv() -> tuple[list[int], list[float]]:
    steps, vals = [], []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            vals.append(float(row["val_loss"]))
    return steps, vals


def main() -> None:
    steps, vals = read_csv()

    min_idx = vals.index(min(vals))
    min_step, min_val = steps[min_idx], vals[min_idx]
    final_step, final_val = steps[-1], vals[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, vals, linestyle="--", marker="o", markersize=3, label="val")
    ax.axvline(
        min_step,
        color="red",
        linestyle=":",
        alpha=0.6,
        label=f"min val @ step {min_step:,} (val={min_val:.2f})",
    )
    ax.axhline(final_val, color="grey", linestyle=":", alpha=0.4)
    ax.annotate(
        f"final val={final_val:.2f}\n(+{(final_val - min_val) / min_val * 100:.0f}% от min)",
        xy=(final_step, final_val),
        xytext=(-100, -40),
        textcoords="offset points",
        fontsize=9,
        ha="right",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("combined NLL (val)")
    ax.set_title("BebopNet — train/val loss (val only; train loss не логируется)")
    ax.legend()
    ax.grid(alpha=0.3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
