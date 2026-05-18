"""Post-processing для генерационных артефактов comparation-pipeline.

Утилиты, перенесённые из удалённого pre-redesign кода (33cfe46^):

- `slice_score`        — копия `cmt_slicing.py::slice_to_chunks`.
- `save_score`         — копия `cmt_slicing.py::save_chunk`.
- `slice_midi`         — копия `slice_gen_into_chunks.py::_slice_into_chunks`.
- `extract_generated`  — копия `extract_generated.py::_split_at_midpoint`.
                         Cut по midpoint downbeats — robust к тому, что
                         модели не всегда возвращают input_bars+output_bars
                         тактов точно (особенно MINGUS где num_chorus =
                         output_bars // input_bars + 1, и реальная длина
                         варьируется).
- `ThemeTooShortError` — копия из `cmt_slicing.py`.
"""
from __future__ import annotations

from pathlib import Path

import music21 as m21
import pretty_midi


class ThemeTooShortError(Exception):
    """Тема короче chunk_bars — для CMT это значит "skip + log"."""


def slice_score(theme_path: Path, chunk_bars: int) -> list[m21.stream.Score]:
    """Возвращает список Score, каждый длиной chunk_bars тактов.

    Бросает ThemeTooShortError если в теме <chunk_bars полных тактов.
    Затакт (Measure с number == 0) попадает в первый чанк через
    score.measures(0, end_bar). Остаток (n % chunk_bars != 0) отбрасывается.
    """
    score = m21.converter.parse(str(theme_path))
    part = score.parts[0]
    full_measures = [m for m in part.getElementsByClass("Measure") if m.number > 0]
    n = len(full_measures)
    if n < chunk_bars:
        raise ThemeTooShortError(
            f"theme has {n} full bars at {theme_path}, need >={chunk_bars}"
        )
    has_pickup = any(m.number == 0 for m in part.getElementsByClass("Measure"))
    n_chunks = n // chunk_bars
    chunks: list[m21.stream.Score] = []
    for i in range(n_chunks):
        start_bar = i * chunk_bars + 1
        end_bar = start_bar + chunk_bars - 1
        effective_start = 0 if (i == 0 and has_pickup) else start_bar
        chunks.append(score.measures(effective_start, end_bar))
    return chunks


def save_score(chunk: m21.stream.Score, path: Path) -> None:
    """Сохраняет Score-чанк в .musicxml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk.write("musicxml", fp=str(path))


def slice_midi(pm: pretty_midi.PrettyMIDI, chunk_bars: int) -> list[pretty_midi.PrettyMIDI]:
    """Режет PrettyMIDI на куски по chunk_bars тактов. Остаток отбрасывается.

    ts/ks: один на старте каждого чанка; не пытаемся сдвигать события из чанка.
    """
    downbeats = pm.get_downbeats()
    n_bars = max(0, len(downbeats) - 1)
    n_chunks = n_bars // chunk_bars
    if n_chunks == 0:
        return []

    chunks: list[pretty_midi.PrettyMIDI] = []
    for i in range(n_chunks):
        start_time = float(downbeats[i * chunk_bars])
        end_time = float(downbeats[(i + 1) * chunk_bars])
        out = pretty_midi.PrettyMIDI(resolution=pm.resolution, initial_tempo=120.0)
        if pm.time_signature_changes:
            ts = pm.time_signature_changes[0]
            out.time_signature_changes.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, 0.0)
            )
        if pm.key_signature_changes:
            ks = pm.key_signature_changes[0]
            out.key_signature_changes.append(
                pretty_midi.KeySignature(ks.key_number, 0.0)
            )
        for ins in pm.instruments:
            new_ins = pretty_midi.Instrument(
                program=ins.program, is_drum=ins.is_drum, name=ins.name,
            )
            for n in ins.notes:
                if start_time - 1e-9 <= n.start < end_time - 1e-9:
                    new_ins.notes.append(pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start - start_time,
                        end=min(n.end, end_time) - start_time,
                    ))
            if new_ins.notes:
                out.instruments.append(new_ins)
        chunks.append(out)
    return chunks


def extract_generated(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Возвращает PrettyMIDI содержащий только сгенерированную часть
    (без префикса темы), сдвинутую к 0.

    Cut по midpoint downbeats — robust к тому, что модели не всегда
    выдают input_bars+output_bars тактов точно. Старый код имел тот
    же подход.
    """
    downbeats = pm.get_downbeats()
    if len(downbeats) < 2:
        return pretty_midi.PrettyMIDI(resolution=pm.resolution, initial_tempo=120.0)
    cut_idx = len(downbeats) // 2
    cut_time = float(downbeats[cut_idx])

    out = pretty_midi.PrettyMIDI(resolution=pm.resolution, initial_tempo=120.0)
    for ts in pm.time_signature_changes:
        if ts.time >= cut_time - 1e-9:
            out.time_signature_changes.append(pretty_midi.TimeSignature(
                ts.numerator, ts.denominator, ts.time - cut_time,
            ))
    if not out.time_signature_changes and pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        out.time_signature_changes.append(
            pretty_midi.TimeSignature(ts.numerator, ts.denominator, 0.0)
        )
    for ks in pm.key_signature_changes:
        if ks.time >= cut_time - 1e-9:
            out.key_signature_changes.append(pretty_midi.KeySignature(
                ks.key_number, ks.time - cut_time,
            ))
    if not out.key_signature_changes and pm.key_signature_changes:
        ks = pm.key_signature_changes[0]
        out.key_signature_changes.append(
            pretty_midi.KeySignature(ks.key_number, 0.0)
        )

    for ins in pm.instruments:
        new_ins = pretty_midi.Instrument(
            program=ins.program, is_drum=ins.is_drum, name=ins.name,
        )
        for n in ins.notes:
            if n.start >= cut_time - 1e-9:
                new_ins.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start - cut_time, end=n.end - cut_time,
                ))
        if new_ins.notes:
            out.instruments.append(new_ins)
    return out
