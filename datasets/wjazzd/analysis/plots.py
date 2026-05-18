"""
WjazzD plotting layer.

One function = one figure. Each function takes a pandas Series/DataFrame and an
output path, renders the plot in a unified visual style, and saves it as PNG.

Style: light theme, sans-serif, slate-blue primary colour, 150 DPI, Russian
labels. Top/right spines are removed and a faint horizontal grid is drawn for
distributions; the heatmap uses its own conventions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PRIMARY = "#4a6a8a"   # slate-blue
ACCENT = "#a04040"    # dark red, used for reference lines / markers
DPI = 150

# Force a sans-serif font that has Cyrillic coverage on macOS/Linux/Windows.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def _setup(figsize: tuple[float, float] = (7.0, 4.2)) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure/axes with our default light style."""
    fig, ax = plt.subplots(figsize=figsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _annotate_stats(ax: plt.Axes, series: pd.Series) -> None:
    """Add median / mean / p95 reference lines + textbox to a histogram axes."""
    median = float(series.median())
    mean = float(series.mean())
    p95 = float(series.quantile(0.95))
    for x, style, label in [
        (median, "-", f"медиана = {median:.1f}"),
        (mean, "--", f"среднее = {mean:.1f}"),
        (p95, ":", f"p95 = {p95:.1f}"),
    ]:
        ax.axvline(x, color=ACCENT, linestyle=style, linewidth=1.2, alpha=0.85, label=label)
    ax.legend(frameon=False, loc="upper right")


# ---------------------------------------------------------------------------
# 1. Notes per solo
# ---------------------------------------------------------------------------

def plot_notes_per_solo(notes: pd.Series, out_path: str | Path) -> None:
    fig, ax = _setup()
    ax.hist(notes, bins=40, color=PRIMARY, edgecolor="white", linewidth=0.6)
    _annotate_stats(ax, notes)
    ax.set_title("Распределение числа нот на соло")
    ax.set_xlabel("Число нот в соло")
    ax.set_ylabel("Количество соло")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Solo duration in bars (with 8 / 16 / 32 markers)
# ---------------------------------------------------------------------------

def plot_solo_duration_bars(duration_bars: pd.Series, out_path: str | Path) -> None:
    fig, ax = _setup()
    ax.hist(duration_bars, bins=40, color=PRIMARY, edgecolor="white", linewidth=0.6)
    median = float(duration_bars.median())
    ax.axvline(median, color=ACCENT, linewidth=1.2, alpha=0.85, label=f"медиана = {median:.0f} тактов")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Распределение длительности соло (в тактах)")
    ax.set_xlabel("Длительность соло, тактов")
    ax.set_ylabel("Количество соло")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Chorus count
# ---------------------------------------------------------------------------

def plot_chorus_count(chorus_count: pd.Series, out_path: str | Path) -> None:
    """Bar chart of chorus_count buckets: 1, 2, 3, ..., 8, >8."""
    counts = chorus_count.dropna().astype(int)
    bucket_labels = ["1", "2", "3", "4", "5", "6", "7", "8", ">8"]
    bucket_values = [
        int((counts == 1).sum()),
        int((counts == 2).sum()),
        int((counts == 3).sum()),
        int((counts == 4).sum()),
        int((counts == 5).sum()),
        int((counts == 6).sum()),
        int((counts == 7).sum()),
        int((counts == 8).sum()),
        int((counts > 8).sum()),
    ]
    fig, ax = _setup()
    bars = ax.bar(bucket_labels, bucket_values, color=PRIMARY, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, bucket_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_title("Распределение числа хорусов на соло")
    ax.set_xlabel("Число хорусов")
    ax.set_ylabel("Количество соло")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Global pitch distribution (MIDI)
# ---------------------------------------------------------------------------

def plot_pitch_distribution(pitches: pd.Series, out_path: str | Path) -> None:
    fig, ax = _setup()
    bins = np.arange(int(pitches.min()), int(pitches.max()) + 2) - 0.5
    ax.hist(pitches, bins=bins, color=PRIMARY, edgecolor="white", linewidth=0.4)
    # Reference octaves on the C notes
    for midi_c, label in [(48, "C3"), (60, "C4 (середина)"), (72, "C5"), (84, "C6")]:
        if pitches.min() <= midi_c <= pitches.max():
            ax.axvline(midi_c, color=ACCENT, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.text(midi_c, ax.get_ylim()[1] * 0.95, label,
                    rotation=90, va="top", ha="right", fontsize=8, color=ACCENT)
    ax.set_title("Глобальное распределение высот нот (MIDI)")
    ax.set_xlabel("MIDI pitch")
    ax.set_ylabel("Количество нотных событий")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Pitch range per solo
# ---------------------------------------------------------------------------

def plot_pitch_range_per_solo(ranges: pd.Series, out_path: str | Path) -> None:
    fig, ax = _setup()
    ax.hist(ranges, bins=range(int(ranges.min()), int(ranges.max()) + 2),
            color=PRIMARY, edgecolor="white", linewidth=0.6)
    _annotate_stats(ax, ranges)
    ax.set_title("Распределение pitch range в соло (полутоны)")
    ax.set_xlabel("Pitch range, полутонов (max − min)")
    ax.set_ylabel("Количество соло")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Note durations (in beats, log-Y)
# ---------------------------------------------------------------------------

def plot_note_durations(durations_beats: pd.Series, out_path: str | Path) -> None:
    """
    Histogram of note durations in fractions of a quarter note.

    The X-axis is clipped to [0, 2] beats — this window contains ~98% of all
    notes in WjazzD; longer notes (held final notes of phrases / solos) are
    very rare and would otherwise force a near-empty tail. Y-axis is linear,
    so the visual height of each bar is directly comparable to the others.
    """
    fig, ax = _setup()
    clip_max = 2.0
    valid = durations_beats[(durations_beats > 0) & (durations_beats <= clip_max)]
    total_all = int(((durations_beats > 0)).sum())
    shown = int(len(valid))
    # Reasonably fine bins (~0.02 beats wide) — narrow enough to resolve the
    # 1/16 / 1/8 / triplet-8th peaks but wide enough to keep individual bars
    # readable on the linear scale.
    ax.hist(valid, bins=100, color=PRIMARY, edgecolor="none")
    # Reference grid for canonical (notated) durations
    references = [
        (1/4, "1/16"),
        (1/3, "тройка 8-й (1/3)"),
        (1/2, "1/8"),
        (2/3, "тройка 1/4 (2/3)"),
        (1.0, "1/4"),
        (1.5, "1/4."),
        (2.0, "1/2"),
    ]
    for x, label in references:
        if 0 < x <= clip_max:
            ax.axvline(x, color=ACCENT, linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(x, ax.get_ylim()[1] * 0.97, label,
                    rotation=90, va="top", ha="right", fontsize=7, color=ACCENT)
    # Honest note about what's NOT shown.
    cut_pct = 100.0 * (total_all - shown) / total_all if total_all else 0.0
    ax.text(0.99, 0.95,
            f"за пределами окна (> {clip_max:g} четв.): {cut_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555")
    ax.set_xlim(0, clip_max)
    ax.set_title("Распределение длительностей нот (в долях четверти)")
    ax.set_xlabel("Длительность ноты, долей четверти")
    ax.set_ylabel("Количество нотных событий")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Melodic intervals (semitones)
# ---------------------------------------------------------------------------

def plot_intervals(intervals: pd.Series, out_path: str | Path) -> None:
    fig, ax = _setup()
    # Clip the long tail beyond +-24 semitones (two octaves) for readability.
    clipped = intervals.clip(lower=-24, upper=24)
    bins = np.arange(-24.5, 25.5, 1)
    ax.hist(clipped, bins=bins, color=PRIMARY, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=ACCENT, linewidth=0.9, alpha=0.6)
    ax.set_title("Распределение мелодических интервалов")
    ax.set_xlabel("Интервал между соседними нотами, полутонов\n(отрицательные = нисходящие, 0 = повтор)")
    ax.set_ylabel("Количество переходов")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 8. Top-20 chord changes
# ---------------------------------------------------------------------------

def plot_top_chords(chord_counts: pd.Series, out_path: str | Path, top_n: int = 20) -> None:
    """Horizontal bar chart of the top-N chords by number of distinct changes."""
    top = chord_counts.head(top_n)[::-1]  # reverse so largest is at the top
    fig, ax = _setup(figsize=(7.0, max(4.0, 0.28 * top_n + 1.0)))
    ax.barh(range(len(top)), top.values, color=PRIMARY, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=8, color="#333")
    ax.set_title(f"Top-{top_n} аккордов в WjazzD (по числу смен аккорда)")
    ax.set_xlabel("Количество смен аккорда")
    # Hide y-grid for horizontal bars; vertical grid is more useful here.
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 9. Heatmap: instrument family x style
# ---------------------------------------------------------------------------

def plot_instrument_family_style_heatmap(xtab: pd.DataFrame, out_path: str | Path) -> None:
    """4 instrument families x 8 styles. Cell value = number of solos."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = xtab.values
    im = ax.imshow(data, aspect="auto", cmap="Blues")
    ax.set_xticks(range(xtab.shape[1]))
    ax.set_xticklabels(xtab.columns, rotation=30, ha="right")
    ax.set_yticks(range(xtab.shape[0]))
    ax.set_yticklabels(xtab.index)
    # Annotate each cell with its count; choose text colour by background brightness.
    vmax = data.max()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            colour = "white" if v > 0.55 * vmax else "#222"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color=colour)
    ax.set_title("Распределение соло по семействам инструментов и стилям")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Количество соло", fontsize=9)
    _save(fig, out_path)
