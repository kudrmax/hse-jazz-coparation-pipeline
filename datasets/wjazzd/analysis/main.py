"""
Run the full WjazzD analysis: load, compute metrics, render 9 figures.

Usage:
    python main.py

Reads ../wjazzd.db, writes 9 PNGs into figures/.
"""

from __future__ import annotations

from pathlib import Path

import data
import plots


HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "wjazzd.db"
FIGURES_DIR = HERE / "figures"


def main() -> None:
    print(f"Reading {DB_PATH}")
    conn = data.connect(DB_PATH)
    try:
        solo_info = data.load_solo_info(conn)
        melody = data.load_melody(conn)
        beats = data.load_beats(conn)
    finally:
        conn.close()
    print(f"Loaded: {len(solo_info)} solos, {len(melody)} note events, {len(beats)} beats")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting 1/9 — notes per solo")
    plots.plot_notes_per_solo(
        data.notes_per_solo(melody),
        FIGURES_DIR / "01_notes_per_solo.png",
    )

    print("Plotting 2/9 — solo duration in bars")
    plots.plot_solo_duration_bars(
        data.solo_duration_bars(melody),
        FIGURES_DIR / "02_solo_duration_bars.png",
    )

    print("Plotting 3/9 — chorus count")
    plots.plot_chorus_count(
        solo_info["chorus_count"],
        FIGURES_DIR / "03_chorus_count.png",
    )

    print("Plotting 4/9 — pitch distribution")
    plots.plot_pitch_distribution(
        data.global_pitches(melody),
        FIGURES_DIR / "04_pitch_distribution.png",
    )

    print("Plotting 5/9 — pitch range per solo")
    plots.plot_pitch_range_per_solo(
        data.pitch_range_per_solo(melody),
        FIGURES_DIR / "05_pitch_range_per_solo.png",
    )

    print("Plotting 6/9 — note durations (log-Y)")
    plots.plot_note_durations(
        data.global_note_durations_in_beats(melody),
        FIGURES_DIR / "06_note_durations.png",
    )

    print("Plotting 7/9 — intervals")
    plots.plot_intervals(
        data.global_intervals(melody),
        FIGURES_DIR / "07_intervals.png",
    )

    print("Plotting 8/9 — top-20 chords")
    plots.plot_top_chords(
        data.chord_vocab(beats),
        FIGURES_DIR / "08_top_chords.png",
    )

    print("Plotting 9/9 — instrument family x style heatmap")
    plots.plot_instrument_family_style_heatmap(
        data.instrument_family_style_xtab(solo_info),
        FIGURES_DIR / "09_instrument_family_style_heatmap.png",
    )

    print(f"Done. Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
