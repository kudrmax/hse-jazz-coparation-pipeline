#!/usr/bin/env python3
"""Build MINGUS val loss curves from epochs.csv.

Run from repo root:
    python pipelines/training-pipeline/scripts/build_mingus_curves.py

Reads:
    pipelines/training-pipeline/results/mingus-wjazzd-paper-optimal-10ep/epochs.csv

Writes:
    pipelines/training-pipeline/figures/mingus-train-val-loss.png

Note: train loss currently NOT logged in epochs.csv (см. metrics.md, статус WAIT:MAYBE).
График строится только по val_loss до решения вопроса о train loss.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = (
    REPO_ROOT
    / "pipelines/training-pipeline/results/mingus-wjazzd-paper-optimal-10ep/epochs.csv"
)
OUT_PATH = REPO_ROOT / "pipelines/training-pipeline/figures/mingus-train-val-loss.png"


def read_phase(phase: str) -> tuple[list[int], list[float]]:
    epochs, vals = [], []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["phase"] == phase:
                epochs.append(int(row["epoch"]))
                vals.append(float(row["val_loss"]))
    return epochs, vals


def main() -> None:
    pitch_ep, pitch_val = read_phase("pitch")
    dur_ep, dur_val = read_phase("duration")

    fig, (ax_p, ax_d) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_p.plot(pitch_ep, pitch_val, linestyle="--", marker="o", label="val_pitch")
    ax_p.set_ylabel("NLL (pitch)")
    ax_p.set_title("MINGUS — train/val loss (val only; train loss не логируется)")
    ax_p.legend()
    ax_p.grid(alpha=0.3)

    ax_d.plot(
        dur_ep,
        dur_val,
        linestyle="--",
        marker="o",
        color="tab:orange",
        label="val_duration",
    )
    ax_d.set_xlabel("epoch")
    ax_d.set_ylabel("NLL (duration)")
    ax_d.legend()
    ax_d.grid(alpha=0.3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
