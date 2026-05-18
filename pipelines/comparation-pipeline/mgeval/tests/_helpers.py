"""Helpers для построения synthetic PrettyMIDI в feature/pipeline тестах."""
from __future__ import annotations

import pretty_midi


def mk_pm(
    notes_spec: list[tuple[int, float, float]],
    *,
    tempo: float = 120.0,
    ts_numerator: int = 4,
    ts_denominator: int = 4,
) -> pretty_midi.PrettyMIDI:
    """Построить PrettyMIDI из списка (pitch, start_sec, end_sec).

    Использует initial_tempo и единственный TimeSignature event.
    Создаёт одну дорожку (program=0).
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.time_signature_changes.append(
        pretty_midi.TimeSignature(ts_numerator, ts_denominator, 0.0)
    )
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes_spec:
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    return pm
