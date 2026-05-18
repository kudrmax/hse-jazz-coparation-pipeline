"""
WjazzD data access + per-solo derived metrics.

Layer 1 of analysis pipeline:
  - Connection helpers and table loaders.
  - Pure functions that compute distributions and aggregates from raw tables.
  - No plotting, no I/O for figures.

All functions are deterministic and operate on pandas DataFrames/Series, so
they can be tested in isolation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants for instrument grouping (used for the 4-family x 8-style heatmap)
# ---------------------------------------------------------------------------

INSTRUMENT_FAMILIES: dict[str, str] = {
    # Saxophones (incl. bass clarinet — historically played by jazz saxophonists)
    "ts": "Saxophones",
    "as": "Saxophones",
    "ss": "Saxophones",
    "bs": "Saxophones",
    "bcl": "Saxophones",
    "ts-c": "Saxophones",
    # Brass
    "tp": "Brass",
    "tb": "Brass",
    "cor": "Brass",
    # Other woodwinds
    "cl": "Other woodwinds",
    # Chordophones / mallets
    "g": "Chordophones & mallets",
    "p": "Chordophones & mallets",
    "vib": "Chordophones & mallets",
}

FAMILY_ORDER: list[str] = [
    "Saxophones",
    "Brass",
    "Other woodwinds",
    "Chordophones & mallets",
]

STYLE_ORDER: list[str] = [
    "TRADITIONAL",
    "SWING",
    "BEBOP",
    "COOL",
    "HARDBOP",
    "POSTBOP",
    "FREE",
    "FUSION",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection to the WjazzD SQLite file."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"WjazzD not found at {db_path}")
    # Read-only URI is overkill here, simple connection is enough.
    return sqlite3.connect(str(db_path))


def load_solo_info(conn: sqlite3.Connection) -> pd.DataFrame:
    """All 456 rows of solo-level metadata."""
    return pd.read_sql_query("SELECT * FROM solo_info", conn)


def load_melody(conn: sqlite3.Connection) -> pd.DataFrame:
    """All ~200k note events with the columns we actually use downstream."""
    cols = ["melid", "eventid", "onset", "pitch", "duration", "bar", "beat", "beatdur"]
    return pd.read_sql_query(
        f"SELECT {', '.join(cols)} FROM melody ORDER BY melid, eventid",
        conn,
    )


def load_beats(conn: sqlite3.Connection) -> pd.DataFrame:
    """All ~132k beat-grid rows with chord annotations."""
    cols = ["melid", "beatid", "bar", "beat", "chord"]
    return pd.read_sql_query(
        f"SELECT {', '.join(cols)} FROM beats ORDER BY melid, beatid",
        conn,
    )


# ---------------------------------------------------------------------------
# Per-solo metrics
# ---------------------------------------------------------------------------

def notes_per_solo(melody: pd.DataFrame) -> pd.Series:
    """Number of note events per solo, indexed by melid."""
    return melody.groupby("melid").size().rename("notes_count")


def solo_duration_bars(melody: pd.DataFrame) -> pd.Series:
    """
    Length of each solo measured in bars (max bar - min bar + 1).

    Note: 'bar' in WjazzD is 1-based per solo; using max-min+1 gives a clean
    integer count of bars spanned by the solo's notes.
    """
    g = melody.groupby("melid")["bar"]
    return (g.max() - g.min() + 1).rename("duration_bars")


def pitch_range_per_solo(melody: pd.DataFrame) -> pd.Series:
    """Span between highest and lowest note in each solo, in semitones."""
    g = melody.groupby("melid")["pitch"]
    return (g.max() - g.min()).rename("pitch_range_semitones").round().astype(int)


# ---------------------------------------------------------------------------
# Global (cross-solo) note-level metrics
# ---------------------------------------------------------------------------

def global_pitches(melody: pd.DataFrame) -> pd.Series:
    """All note pitches in the dataset, rounded to integer MIDI numbers."""
    return melody["pitch"].round().astype(int).rename("pitch")


def global_note_durations_in_beats(melody: pd.DataFrame) -> pd.Series:
    """
    All note durations expressed in fractions of a quarter note (i.e., beats).

    Computed per-event as duration / beatdur, where beatdur is the duration of
    the beat in seconds for the metric context of that event. This is robust to
    tempo changes within a solo, unlike using a single avgtempo.
    """
    durs = melody["duration"] / melody["beatdur"]
    return durs.rename("duration_beats")


def global_intervals(melody: pd.DataFrame) -> pd.Series:
    """
    Melodic intervals (semitones) between consecutive notes within each solo.

    First note of each solo has no predecessor and is dropped. Pitch values are
    rounded to integer semitones before differencing.
    """
    pitches_int = melody.assign(pitch_int=melody["pitch"].round().astype(int))
    pitches_int = pitches_int.sort_values(["melid", "eventid"])
    intervals = pitches_int.groupby("melid")["pitch_int"].diff()
    return intervals.dropna().astype(int).rename("interval_semitones")


# ---------------------------------------------------------------------------
# Chord vocabulary
# ---------------------------------------------------------------------------

def chord_vocab(beats: pd.DataFrame) -> pd.Series:
    """
    Counts of unique chord changes across the whole dataset.

    A 'chord change' = a beat whose chord differs from the previous beat in
    the same solo. This deduplicates runs of consecutive beats holding the
    same chord. The first beat of every solo also counts (no predecessor).

    Empty/whitespace-only chord labels and NaN are dropped.
    """
    df = beats.sort_values(["melid", "beatid"]).copy()
    df["prev_chord"] = df.groupby("melid")["chord"].shift()
    is_change = df["chord"] != df["prev_chord"]
    changes = df.loc[is_change, "chord"].dropna()
    changes = changes[changes.astype(str).str.strip() != ""]
    return changes.value_counts().rename("count")


# ---------------------------------------------------------------------------
# Cross-tabulation
# ---------------------------------------------------------------------------

def instrument_family_style_xtab(solo_info: pd.DataFrame) -> pd.DataFrame:
    """
    4-family x 8-style cross-tabulation of solo counts.

    Solos with an instrument code not in INSTRUMENT_FAMILIES are dropped; this
    should not happen for WjazzD v2.x but is defensive. Rows and columns are
    reindexed to the canonical orders defined at the top of this module.
    """
    df = solo_info.copy()
    df["family"] = df["instrument"].map(INSTRUMENT_FAMILIES)
    df = df.dropna(subset=["family"])
    xt = pd.crosstab(df["family"], df["style"])
    return xt.reindex(index=FAMILY_ORDER, columns=STYLE_ORDER, fill_value=0)
