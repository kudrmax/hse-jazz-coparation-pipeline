"""Base Input / Output dataclasses for the generation pipeline.

Concrete wrappers (Mingus / Bebopnet / Cmt) inherit these and append
model-specific fields. Comparation-pipeline operates on the base view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import music21 as m21
import pretty_midi


@dataclass
class BaseGeneratorInput:
    musicxml_path: Path
    seed: int
    input_bars: int
    output_bars: int

    def get_musicxml_path(self) -> Path:
        return self.musicxml_path


@dataclass
class BaseGeneratorOutput:
    midi: pretty_midi.PrettyMIDI
    title: str
    seed: int
    input_bars: int
    output_bars: int
    inference_time: float
    # Captured by CommonPostProcessor from the source theme: list of
    # (offset_in_quarters, ChordSymbol object) for every ChordSymbol
    # element. Stored as full m21 objects (not just figure strings)
    # because m21 cannot reliably re-parse extended figures like
    # "Eb7add b9". Used by save_musicxml_with_chords to stamp them
    # onto the output cyclically across the full length.
    theme_chord_symbols: list[tuple[float, m21.harmony.ChordSymbol]] = field(
        default_factory=list
    )

    def get_midi(self) -> pretty_midi.PrettyMIDI:
        return self.midi

    def save_midi(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.midi.write(str(path))
        return path

    def save_musicxml(self, path: Path) -> Path:
        """Write self.midi as melody-only MusicXML."""
        from .midi_to_musicxml import MidiToMusicxmlConverter

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        score = MidiToMusicxmlConverter.to_melody_only(self.midi)
        actual = Path(str(score.write("musicxml", fp=str(path))))
        if actual != path:
            actual.replace(path)
        return path

    def save_musicxml_with_chords(self, path: Path) -> Path:
        """Write self.midi as MusicXML with the theme's chord symbols
        overlaid cyclically across the full input + output length."""
        from .midi_to_musicxml import MidiToMusicxmlConverter

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        score = MidiToMusicxmlConverter.to_with_chords(
            self.midi,
            theme_chord_symbols=self.theme_chord_symbols,
            input_bars=self.input_bars,
            output_bars=self.output_bars,
        )
        actual = Path(str(score.write("musicxml", fp=str(path))))
        if actual != path:
            actual.replace(path)
        return path
