"""Tests for CommonPostProcessor — sets the key_signature meta-event."""
from pathlib import Path

import music21 as m21
import pretty_midi

from models.base.io import BaseGeneratorInput, BaseGeneratorOutput
from models.base.post_processor import CommonPostProcessor


def _build_a_minor_stream(n_bars: int = 8) -> m21.stream.Score:
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.key.Key("A", "minor"))
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        # Notes from a-minor scale so analyze('key') is unambiguous.
        for pitch in ("A4", "C5", "E5", "B4"):
            m.append(m21.note.Note(pitch, quarterLength=1.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    return score


def _build_c_major_stream(n_bars: int = 8) -> m21.stream.Score:
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    part.append(m21.key.Key("C", "major"))
    for bar in range(n_bars):
        m = m21.stream.Measure(number=bar + 1)
        for pitch in ("C4", "E4", "G4", "C5"):
            m.append(m21.note.Note(pitch, quarterLength=1.0))
        part.append(m)
    score = m21.stream.Score()
    score.append(part)
    return score


def _make_output() -> BaseGeneratorOutput:
    return BaseGeneratorOutput(
        midi=pretty_midi.PrettyMIDI(),
        title="x",
        seed=1,
        input_bars=8,
        output_bars=8,
        inference_time=0.0,
    )


def test_appends_one_key_signature():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    assert out.midi.key_signature_changes == []

    CommonPostProcessor().process(inp, stream, out)

    assert len(out.midi.key_signature_changes) == 1
    ks = out.midi.key_signature_changes[0]
    assert ks.time == 0.0


def test_a_minor_maps_to_key_number_21():
    """pretty_midi key_number convention: 0..11 = major C..B; 12..23 = minor C..B.
    A minor → 12 + 9 = 21."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    CommonPostProcessor().process(inp, stream, out)
    assert out.midi.key_signature_changes[0].key_number == 21


def test_tempo_from_theme_metronome_mark():
    """If the theme has a MetronomeMark, its BPM ends up on the output MIDI."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    stream.parts[0].insert(0, m21.tempo.MetronomeMark(number=90))

    out = _make_output()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    out.midi.instruments.append(inst)

    CommonPostProcessor().process(inp, stream, out)

    times, tempos = out.midi.get_tempo_changes()
    assert len(tempos) == 1
    assert abs(tempos[0] - 90.0) < 0.01


def test_tempo_default_when_theme_has_no_metronome_mark():
    """If the theme has no MetronomeMark, DEFAULT_TEMPO_BPM (180) is applied."""
    from models.base.post_processor import DEFAULT_TEMPO_BPM

    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()  # no MetronomeMark inserted
    out = _make_output()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    out.midi.instruments.append(inst)

    CommonPostProcessor().process(inp, stream, out)

    times, tempos = out.midi.get_tempo_changes()
    assert len(tempos) == 1
    assert abs(tempos[0] - DEFAULT_TEMPO_BPM) < 0.01


def test_tempo_change_rescales_note_durations():
    """A note that was 1 quarter at 120 BPM (=0.5s) must still be 1 quarter
    after the post-processor moves the tempo to 60 BPM (=1.0s)."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    stream.parts[0].insert(0, m21.tempo.MetronomeMark(number=60))

    out = _make_output()
    out.midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    out.midi.instruments.append(inst)

    CommonPostProcessor().process(inp, stream, out)

    note = out.midi.instruments[0].notes[0]
    # 60 BPM is half as fast as 120 → durations double in seconds, but the
    # note still spans exactly one quarter musically.
    assert abs(note.end - 1.0) < 0.001


def test_writes_4_4_time_signature():
    """Pipeline is 4/4-only — output must declare it explicitly,
    not rely on whatever default the underlying MIDI library happens to use."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    assert out.midi.time_signature_changes == []

    CommonPostProcessor().process(inp, stream, out)

    assert len(out.midi.time_signature_changes) == 1
    ts = out.midi.time_signature_changes[0]
    assert ts.numerator == 4
    assert ts.denominator == 4
    assert ts.time == 0.0


def test_replaces_existing_time_signature():
    """Replace, not append, so we don't end up with two events."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    out.midi.time_signature_changes.append(
        pretty_midi.TimeSignature(numerator=3, denominator=4, time=0.0)
    )
    CommonPostProcessor().process(inp, stream, out)
    assert len(out.midi.time_signature_changes) == 1
    assert out.midi.time_signature_changes[0].numerator == 4


def test_normalizes_instrument_program_to_piano():
    """All output instruments must end up on program 0 (Acoustic Grand Piano)
    so editors don't apply transposing-instrument notation shifts (e.g.
    program 67 = Baritone Sax adds +3 sharps in MuseScore's view)."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    sax = pretty_midi.Instrument(program=67, name="Tenor Sax")
    sax.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    flute = pretty_midi.Instrument(program=73, name="Flute")
    flute.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.0, end=0.5))
    out.midi.instruments.extend([sax, flute])

    CommonPostProcessor().process(inp, stream, out)

    for instrument in out.midi.instruments:
        assert instrument.program == 0
        assert instrument.name == "Piano"


def test_captures_theme_chord_symbols():
    """Post-processor stashes the theme's ChordSymbols onto the output for
    later save_musicxml_with_chords to consume. We snapshot them as
    full m21 objects (not figure strings)."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    stream.parts[0].insert(0.0, m21.harmony.ChordSymbol("Am"))
    stream.parts[0].insert(8.0, m21.harmony.ChordSymbol("Dm"))
    out = _make_output()

    CommonPostProcessor().process(inp, stream, out)

    assert len(out.theme_chord_symbols) == 2
    offsets = [t[0] for t in out.theme_chord_symbols]
    figures = [t[1].figure for t in out.theme_chord_symbols]
    assert offsets == [0.0, 8.0]
    assert figures == ["Am", "Dm"]
    assert all(isinstance(t[1], m21.harmony.ChordSymbol) for t in out.theme_chord_symbols)


def test_replaces_existing_key_signature():
    """When upstream exporter (e.g. music21 stream.write) already wrote a
    key_signature, post_processor must REPLACE it (not append), so the
    final MIDI carries exactly one key_signature — ours."""
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_a_minor_stream()
    out = _make_output()
    out.midi.key_signature_changes.append(
        pretty_midi.KeySignature(key_number=0, time=0.0)  # bogus pre-existing
    )
    CommonPostProcessor().process(inp, stream, out)
    assert len(out.midi.key_signature_changes) == 1
    assert out.midi.key_signature_changes[0].key_number == 21  # a minor, not 0


def test_c_major_maps_to_key_number_0():
    inp = BaseGeneratorInput(
        musicxml_path=Path("/tmp/x.xml"), seed=1, input_bars=8, output_bars=8
    )
    stream = _build_c_major_stream()
    out = _make_output()
    CommonPostProcessor().process(inp, stream, out)
    assert out.midi.key_signature_changes[0].key_number == 0
