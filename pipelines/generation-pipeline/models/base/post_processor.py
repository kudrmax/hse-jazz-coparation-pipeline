"""Common post-processing applied to every generator's output MIDI.

Per-model subclasses inherit this class and call super().process(...)
before adding their own steps (Z-style vertical inheritance — same
shape as CommonInputValidator → *InputValidator).

`process()` is a thin orchestrator — each independent step lives in
its own private method so a future reader can find them by name.
"""
from __future__ import annotations

import music21 as m21
import pretty_midi

from .io import BaseGeneratorInput, BaseGeneratorOutput

NORMALIZED_PROGRAM = 0  # GM Acoustic Grand Piano — non-transposing default
NORMALIZED_INSTRUMENT_NAME = "Piano"
PIPELINE_TIME_SIGNATURE_NUMERATOR = 4
PIPELINE_TIME_SIGNATURE_DENOMINATOR = 4
DEFAULT_TEMPO_BPM = 180.0  # used when the theme xml has no MetronomeMark


def _music21_key_to_pretty_midi_number(key: m21.key.Key) -> int:
    """pretty_midi key_number convention: 0..11 = major C..B; 12..23 = minor C..B."""
    return key.tonic.pitchClass + (12 if key.mode == "minor" else 0)


def _read_tempo_from_theme_or_default(parsed_stream: m21.stream.Score) -> float:
    """Return the first MetronomeMark.number from the theme, or DEFAULT_TEMPO_BPM
    if the theme has no metronome marking. Pure read, no side effects."""
    marks = list(parsed_stream.recurse().getElementsByClass(m21.tempo.MetronomeMark))
    if marks and marks[0].number is not None:
        return float(marks[0].number)
    return DEFAULT_TEMPO_BPM


def _current_tempo_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    """Read the first set_tempo event from the PrettyMIDI as BPM."""
    times, tempos = midi.get_tempo_changes()
    if len(tempos) == 0:
        return DEFAULT_TEMPO_BPM
    return float(tempos[0])


def _retempo_pretty_midi(
    midi: pretty_midi.PrettyMIDI, target_bpm: float
) -> pretty_midi.PrettyMIDI:
    """Return a fresh PrettyMIDI whose initial_tempo is `target_bpm` and whose
    notes were rescaled so they fall on the same rhythmic positions
    (i.e. a quarter-note stays a quarter-note, just played slower/faster).

    pretty_midi.PrettyMIDI has no public setter for tempo — the field is
    only honoured at construction time. We rebuild the container around
    the existing instruments, scaling note start/end by the BPM ratio.
    """
    current_bpm = _current_tempo_bpm(midi)
    ratio = current_bpm / target_bpm

    new_midi = pretty_midi.PrettyMIDI(initial_tempo=target_bpm)
    for instrument in midi.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            new_instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start * ratio,
                    end=note.end * ratio,
                )
            )
        new_midi.instruments.append(new_instrument)
    new_midi.key_signature_changes = list(midi.key_signature_changes)
    new_midi.time_signature_changes = list(midi.time_signature_changes)
    return new_midi


class CommonPostProcessor:
    def process(
        self,
        inp: BaseGeneratorInput,
        parsed_stream: m21.stream.Score,
        output: BaseGeneratorOutput,
    ) -> None:
        self._set_tempo_from_theme_or_default(parsed_stream, output)
        self._set_key_signature_from_theme(parsed_stream, output)
        self._set_time_signature_to_pipeline_default(output)
        self._normalize_instruments_to_piano(output)
        self._capture_theme_chord_symbols(parsed_stream, output)

    def _set_tempo_from_theme_or_default(
        self,
        parsed_stream: m21.stream.Score,
        output: BaseGeneratorOutput,
    ) -> None:
        """Stamp the theme's MetronomeMark BPM (or DEFAULT_TEMPO_BPM if absent)
        onto the output MIDI. Note start/end are rescaled so rhythmic
        positions stay correct regardless of the source models' default."""
        target_bpm = _read_tempo_from_theme_or_default(parsed_stream)
        output.midi = _retempo_pretty_midi(output.midi, target_bpm=target_bpm)

    def _set_key_signature_from_theme(
        self,
        parsed_stream: m21.stream.Score,
        output: BaseGeneratorOutput,
    ) -> None:
        """Stamp the analyzed key of the source theme onto the output MIDI.

        Without this, downstream editors guess a key from note distribution
        and frequently pick something far from the theme's tonality.
        Replaces (does not append) — some upstream exporters already emit
        a key_signature meta-event, and we want exactly one.
        """
        key = parsed_stream.analyze("key")
        key_number = _music21_key_to_pretty_midi_number(key)
        output.midi.key_signature_changes = [
            pretty_midi.KeySignature(key_number=key_number, time=0.0)
        ]

    def _set_time_signature_to_pipeline_default(
        self, output: BaseGeneratorOutput
    ) -> None:
        """Stamp 4/4 onto the output MIDI explicitly.

        Without this, the time_signature meta-event in the output is whatever
        the underlying MIDI library defaults to (which happens to be 4/4 for
        pretty_midi/mido — but that's incidental, not a guarantee from the
        models). Since the pipeline as a whole only validates 4/4 themes,
        we make 4/4 explicit on the output as well.
        """
        output.midi.time_signature_changes = [
            pretty_midi.TimeSignature(
                numerator=PIPELINE_TIME_SIGNATURE_NUMERATOR,
                denominator=PIPELINE_TIME_SIGNATURE_DENOMINATOR,
                time=0.0,
            )
        ]

    def _capture_theme_chord_symbols(
        self,
        parsed_stream: m21.stream.Score,
        output: BaseGeneratorOutput,
    ) -> None:
        """Snapshot the theme's ChordSymbol elements as
        (offset_q, ChordSymbol-object) pairs onto the output. Used later
        by save_musicxml_with_chords to overlay them cyclically.

        Stored as full m21 objects (deep-copied) rather than figure strings
        because m21 cannot reliably re-parse extended figures like
        "Eb7 add b9" — round-trip through ChordSymbol(figure) raises.
        """
        import copy

        captured: list[tuple[float, m21.harmony.ChordSymbol]] = []
        for cs in parsed_stream.recurse().getElementsByClass(m21.harmony.ChordSymbol):
            offset_q = float(cs.getOffsetInHierarchy(parsed_stream))
            captured.append((offset_q, copy.deepcopy(cs)))
        output.theme_chord_symbols = captured

    def _normalize_instruments_to_piano(self, output: BaseGeneratorOutput) -> None:
        """Force every instrument to GM program 0 (non-transposing piano).

        Wrappers ship different GM programs by default (CMT/BebopNet: 0;
        MINGUS: 67 = Baritone Sax, an Eb transposing instrument). Editors
        that honour transposing-instrument conventions render the latter
        with a notation shift (e.g. +3 sharps in MuseScore). Forcing
        program 0 removes that source of visual drift across the three
        models without changing audio.
        """
        for instrument in output.midi.instruments:
            instrument.program = NORMALIZED_PROGRAM
            instrument.name = NORMALIZED_INSTRUMENT_NAME
