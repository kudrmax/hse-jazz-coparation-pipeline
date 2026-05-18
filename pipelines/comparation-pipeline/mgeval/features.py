"""Feature extraction — порт mgeval/core.py.

9 публичных функций (через FEATURES registry в конце файла), все принимают
pretty_midi.PrettyMIDI одного 8-тактового куска (monophonic).
Возвращают np.ndarray (scalar → (1,), histogram → (12,), matrix → flatten (144,)).
Возвращают None если кусок дегенеративен для данной фичи (len(notes) == 0
или для парных фич len(notes) < 2).
"""
from __future__ import annotations

import numpy as np
import pretty_midi


# === Private helpers ===

_DEFAULT_TEMPO = 120.0
_DEFAULT_TS_NUMERATOR = 4
_DEFAULT_TS_DENOMINATOR = 4


def _get_sorted_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    """Все ноты со всех инструментов, отсортированы по start (и pitch для стабильности)."""
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _bar_length_seconds(pm: pretty_midi.PrettyMIDI) -> float:
    """Длина бара в секундах. Tempo: первый явный tempo event (fallback 120 BPM).
    Time signature: первый ts event (fallback 4/4).
    """
    tempo_times, tempi = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) > 0 else _DEFAULT_TEMPO
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        numer = ts.numerator
        denom = ts.denominator
    else:
        numer = _DEFAULT_TS_NUMERATOR
        denom = _DEFAULT_TS_DENOMINATOR
    return (60.0 / tempo) * numer * (4.0 / denom)


# === Public feature functions ===


def total_used_pitch(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """Число уникальных pitch'ей в куске. Reference: total_used_pitch."""
    notes = _get_sorted_notes(pm)
    if len(notes) == 0:
        return None
    return np.array([len({n.pitch for n in notes})], dtype=float)


def total_used_note(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """Число нот в куске (pitch-агностично). Reference: total_used_note."""
    notes = _get_sorted_notes(pm)
    if len(notes) == 0:
        return None
    return np.array([len(notes)], dtype=float)


def _safe_piano_roll(pm: pretty_midi.PrettyMIDI):
    """piano_roll первого инструмента или None если нет инструментов.

    Reference использует instruments[0]; у нас single-instrument после
    postprocess normalize, но защита от пустых instruments всё равно нужна.
    """
    if not pm.instruments:
        return None
    return pm.instruments[0].get_piano_roll(fs=100)


def pitch_range(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """Разница max-min pitch в полутонах. Reference: pitch_range (через piano_roll)."""
    notes = _get_sorted_notes(pm)
    if len(notes) == 0:
        return None
    piano_roll = _safe_piano_roll(pm)
    if piano_roll is None:
        return None
    pitch_indices = np.where(np.sum(piano_roll, axis=1) > 0)[0]
    if len(pitch_indices) == 0:
        return None
    return np.array([int(np.max(pitch_indices) - np.min(pitch_indices))], dtype=float)


def total_pitch_class_histogram(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """12-вектор pitch-class распределения, нормализован на sum=1.
    Reference: total_pitch_class_histogram (через piano_roll, duration-weighted).
    """
    notes = _get_sorted_notes(pm)
    if len(notes) == 0:
        return None
    piano_roll = _safe_piano_roll(pm)
    if piano_roll is None:
        return None
    histogram = np.zeros(12)
    sums_per_pitch = np.sum(piano_roll, axis=1)
    for pitch in range(128):
        histogram[pitch % 12] += sums_per_pitch[pitch]
    total = histogram.sum()
    if total == 0:
        return None
    return histogram / total


def pitch_class_transition_matrix(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """12×12 матрица переходов pitch-class. Reference: normalize=0 (raw counts).
    Возвращается flatten в shape (144,) для евклид-расстояния.
    """
    notes = _get_sorted_notes(pm)
    if len(notes) < 2:
        return None
    matrix = pm.get_pitch_class_transition_matrix()
    return matrix.flatten().astype(float)


def avg_pitch_interval(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """mean(|interval|) между соседними нотами в полутонах.
    Reference: avg_pitch_shift (несмотря на docstring «average value», impl — abs).
    Также называется «Pitch Interval (PI)» в CMT-paper.
    """
    notes = _get_sorted_notes(pm)
    if len(notes) < 2:
        return None
    pitches = np.array([n.pitch for n in notes])
    intervals = np.diff(pitches)
    return np.array([float(np.mean(np.abs(intervals)))], dtype=float)


def avg_ioi(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """Среднее inter-onset-interval в секундах. Reference: avg_IOI (через onsets)."""
    notes = _get_sorted_notes(pm)
    if len(notes) < 2:
        return None
    onsets = np.array([n.start for n in notes])
    iois = np.diff(onsets)
    return np.array([float(np.mean(iois))], dtype=float)


NLH_BINS_RATIO = [96, 48, 24, 12, 6, 72, 36, 18, 9, 32, 16, 8]
# индексы → длительности (в единицах bar_length/96):
# 0: full(96), 1: half(48), 2: quarter(24), 3: 8th(12), 4: 16th(6),
# 5: dot-half(72), 6: dot-quarter(36), 7: dot-8th(18), 8: dot-16th(9),
# 9: half-triplet(32), 10: quarter-triplet(16), 11: 8th-triplet(8)


def _note_length_bin_idx(length_seconds: float, unit_seconds: float) -> int:
    hist_list = np.array([unit_seconds * r for r in NLH_BINS_RATIO])
    return int(np.argmin(np.abs(hist_list - length_seconds)))


def note_length_hist(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """Гистограмма длительностей нот по 12 фиксированным бинам, нормализована на sum=1.
    Reference: note_length_hist(pause_event=False).

    Конвертация: bar_length_seconds → unit_seconds = bar_length/96 → bin =
    argmin(|hist_list_seconds - length_seconds|).
    """
    notes = _get_sorted_notes(pm)
    if len(notes) == 0:
        return None
    unit = _bar_length_seconds(pm) / 96.0
    hist = np.zeros(12)
    for n in notes:
        idx = _note_length_bin_idx(n.end - n.start, unit)
        hist[idx] += 1
    total = hist.sum()
    if total == 0:
        return None
    return hist / total


def note_length_transition_matrix(pm: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """12×12 матрица переходов между категориями длительностей.
    Reference: note_length_transition_matrix(pause_event=False, normalize=0).
    Возвращается flatten в shape (144,).
    """
    notes = _get_sorted_notes(pm)
    if len(notes) < 2:
        return None
    unit = _bar_length_seconds(pm) / 96.0
    matrix = np.zeros((12, 12))
    last_idx: int | None = None
    for n in notes:
        cur_idx = _note_length_bin_idx(n.end - n.start, unit)
        if last_idx is not None:
            matrix[last_idx][cur_idx] += 1
        last_idx = cur_idx
    return matrix.flatten()


# === Public registry (порядок зафиксирован в spec) ===

FEATURES = {
    "total_used_pitch": total_used_pitch,
    "total_pitch_class_histogram": total_pitch_class_histogram,
    "pitch_class_transition_matrix": pitch_class_transition_matrix,
    "pitch_range": pitch_range,
    "avg_pitch_interval": avg_pitch_interval,
    "total_used_note": total_used_note,
    "avg_ioi": avg_ioi,
    "note_length_hist": note_length_hist,
    "note_length_transition_matrix": note_length_transition_matrix,
}
