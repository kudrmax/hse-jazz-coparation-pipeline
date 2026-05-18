"""Convert PrettyMIDI outputs into music21 Score for MusicXML export.

The MIDI standard does not carry chord symbols (no `<harmony>` equivalent),
so editors that consume our `.mid` outputs lose the lead-sheet-level
chord-name information from the source theme. MusicXML preserves it via
`<harmony>` elements; this helper converts our generated MIDI to an
m21 Score and optionally overlays the source theme's chord symbols
across the full output, repeating them cyclically over the
input + output bars (which is exactly the harmonic context every wrapper
trains and infers under).

Stateless: no class state, no inheritance — used the same way for all
three wrappers.

The Score is built directly from PrettyMIDI via the m21 API — no
midi-tempfile roundtrip. We work in integer 1/24-quarter units throughout,
splitting each note across bar AND quarter boundaries with ties. Each
quarter is rendered in one of two modes:

  - triplet  — at least one note position (start or end) lies on the
               8th-triplet grid but not on the 16th grid. The whole
               quarter is bracketed as a full 8th-triplet group of three
               slots {0, 8, 16}; absent slots are filled with rests.
  - power-of-2 — otherwise. Notes snap to the 16th grid (multiples of
               6 within the quarter); decomposition uses standard
               power-of-2 + dotted durations only.

After per-quarter rendering, a final pass merges fully-closed tied chains
of same-pitch notes into one note where the total duration is a clean
standard duration (whole, dotted-half, half, etc.) — purely cosmetic, the
underlying timing is unchanged.
"""
from __future__ import annotations

import copy
import math
from fractions import Fraction

import music21 as m21
import pretty_midi

DEFAULT_BEATS_PER_BAR = 4

# Smallest common denominator across the three models' note grids:
#   - CMT     (preprocess.py:44)            → 1/4 quarter (16th)         = 6/24
#   - MINGUS  (gen_funct.py:39)             → 1/8, 1/12 quarter          = 3/24, 2/24
#   - BebopNet(gather_data_from_xml.py:249) → 1/12 quarter (12 per beat) = 2/24
UNITS_PER_QUARTER = 24
UNITS_PER_BAR = DEFAULT_BEATS_PER_BAR * UNITS_PER_QUARTER  # 96

TRIPLET_POSITIONS_IN_QUARTER = (0, 8, 16)
TRIPLET_SLOT_UNITS = 8

# Power-of-2 + dotted notatable durations (in 1/24 quarter units), without
# triplet members. Greedy-decomposable for any positive multiple of 3.
P2_NOTATABLE_24THS = (96, 72, 48, 36, 24, 18, 12, 9, 6, 3)

# Inside-quarter positions/ends that hint at the active grid.
_TRIPLET_HINT = {8, 16}
_P2_HINT = {3, 6, 9, 12, 15, 18, 21}

_P2_POSITIONS_IN_QUARTER = (0, 3, 6, 9, 12, 15, 18, 21, 24)
_TRIPLET_POSITIONS_FOR_SNAP = (0, 8, 16, 24)


def _decompose_p2(units: int) -> list[int]:
    pieces: list[int] = []
    remaining = units
    while remaining > 0:
        for v in P2_NOTATABLE_24THS:
            if v <= remaining:
                pieces.append(v)
                remaining -= v
                break
        else:
            break
    return pieces


def _ties_for_pieces(num_pieces: int, outer_tie: str | None) -> list[str | None]:
    if num_pieces == 1:
        return [outer_tie]
    if outer_tie is None:
        return ["start"] + ["continue"] * (num_pieces - 2) + ["stop"]
    if outer_tie == "start":
        return ["start"] + ["continue"] * (num_pieces - 1)
    if outer_tie == "continue":
        return ["continue"] * num_pieces
    if outer_tie == "stop":
        return ["continue"] * (num_pieces - 1) + ["stop"]
    raise ValueError(f"unknown tie state: {outer_tie}")


def _split_into_quarter_fragments(start_u: int, end_u: int):
    """Yield (bar_idx, quarter_idx_in_bar, lo_in_q, ld_in_q, tie_state)
    fragments of one note. Quarter boundaries occur every 24 units, bar
    boundaries every 96 units. Tie state covers cross-quarter joins."""
    cur = start_u
    while cur < end_u:
        bar_idx = cur // UNITS_PER_BAR
        q_idx_in_bar = (cur % UNITS_PER_BAR) // UNITS_PER_QUARTER
        q_start = bar_idx * UNITS_PER_BAR + q_idx_in_bar * UNITS_PER_QUARTER
        q_end = q_start + UNITS_PER_QUARTER
        local_offset = cur - q_start
        local_end = min(end_u, q_end)
        local_dur = local_end - cur

        is_first = cur == start_u
        is_last = local_end >= end_u
        if is_first and is_last:
            tie = None
        elif is_first:
            tie = "start"
        elif is_last:
            tie = "stop"
        else:
            tie = "continue"

        yield (bar_idx, q_idx_in_bar, local_offset, local_dur, tie)
        cur = local_end


def _collect_notes_in_units(midi: pretty_midi.PrettyMIDI) -> list[tuple[int, int, int]]:
    """Return [(start_u, end_u, pitch)] in integer 1/24-quarter units,
    monophonically clipped. Rounding to integer units is the snap step."""
    times, tempos = midi.get_tempo_changes()
    bpm = float(tempos[0]) if len(tempos) else 120.0
    sec_per_quarter = 60.0 / bpm

    raw: list[tuple[int, int, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            start_u = int(round((note.start / sec_per_quarter) * UNITS_PER_QUARTER))
            end_u = int(round((note.end / sec_per_quarter) * UNITS_PER_QUARTER))
            if end_u <= start_u:
                end_u = start_u + 1
            raw.append((start_u, end_u, note.pitch))
    raw.sort(key=lambda t: (t[0], t[1]))

    cleaned: list[tuple[int, int, int]] = []
    for i, (s, e, p) in enumerate(raw):
        if i + 1 < len(raw):
            next_s = raw[i + 1][0]
            if e > next_s:
                e = next_s
        if e <= s:
            continue
        cleaned.append((s, e, p))
    return cleaned


def _read_metadata(midi: pretty_midi.PrettyMIDI) -> tuple[float, m21.key.Key | None]:
    times, tempos = midi.get_tempo_changes()
    bpm = float(tempos[0]) if len(tempos) else 120.0
    key: m21.key.Key | None = None
    if midi.key_signature_changes:
        ks = midi.key_signature_changes[0]
        if ks.key_number < 12:
            tonic_pc, mode = ks.key_number, "major"
        else:
            tonic_pc, mode = ks.key_number - 12, "minor"
        tonic_name = m21.pitch.Pitch(midi=60 + tonic_pc).name
        key = m21.key.Key(tonic_name, mode)
    return bpm, key


def _quarter_mode(events: list[tuple[int, int, int, str | None]]) -> str:
    """Return 'triplet' or 'p2' based on positions/ends inside the quarter."""
    has_triplet = False
    has_p2 = False
    for lo, ld, _pitch, _tie in events:
        end = lo + ld
        for pos in (lo, end):
            if pos in _TRIPLET_HINT:
                has_triplet = True
            if pos in _P2_HINT:
                has_p2 = True
    if has_triplet and not has_p2:
        return "triplet"
    return "p2"


def _snap_to(positions: tuple[int, ...], u: int) -> int:
    return min(positions, key=lambda p: abs(p - u))


def _snap_events_to_mode(events, mode: str):
    """Snap (lo, ld) to the mode's grid. Drop events that collapse to zero
    duration after the snap. Clip overlaps so consecutive notes in one
    quarter never overlap."""
    grid = (
        _TRIPLET_POSITIONS_FOR_SNAP if mode == "triplet" else _P2_POSITIONS_IN_QUARTER
    )
    snapped = []
    for lo, ld, pitch, tie in events:
        new_lo = _snap_to(grid, lo)
        new_end = _snap_to(grid, lo + ld)
        if new_end <= new_lo:
            continue
        snapped.append((new_lo, new_end - new_lo, pitch, tie))
    snapped.sort(key=lambda t: t[0])
    deduped: list[tuple[int, int, int, str | None]] = []
    for ev in snapped:
        if deduped and deduped[-1][0] == ev[0]:
            continue
        if deduped:
            prev_lo, prev_ld, prev_pitch, prev_tie = deduped[-1]
            prev_end = prev_lo + prev_ld
            if prev_end > ev[0]:
                deduped[-1] = (prev_lo, ev[0] - prev_lo, prev_pitch, prev_tie)
        deduped.append(ev)
    deduped = [e for e in deduped if e[1] > 0]
    return deduped


def _make_triplet_element(
    piece_units: int, pitch_or_none: int | None, bracket_pos: str
) -> m21.note.GeneralNote:
    qL = Fraction(piece_units, UNITS_PER_QUARTER)
    if pitch_or_none is None:
        el = m21.note.Rest(quarterLength=qL)
    else:
        el = m21.note.Note(pitch_or_none, quarterLength=qL)
    t = m21.duration.Tuplet(
        numberNotesActual=3, numberNotesNormal=2, durationNormal="eighth"
    )
    if bracket_pos == "start":
        t.type = "start"
    elif bracket_pos == "stop":
        t.type = "stop"
    el.duration.appendTuplet(t)
    return el


def _build_triplet_quarter(
    events, measure: m21.stream.Measure, quarter_offset_in_bar: int
):
    """Render a triplet-mode quarter: 3 fixed slots {0, 8, 16}, each is
    either a note (possibly tied across slots) or a rest. All three sit
    inside one 8th-triplet bracket."""
    slots: list[tuple[int | None, str | None]] = []
    for slot_offset in TRIPLET_POSITIONS_IN_QUARTER:
        active_pitch: int | None = None
        active_tie: str | None = None
        for lo, ld, pitch, tie in events:
            if lo <= slot_offset < lo + ld:
                active_pitch = pitch
                slot_is_first = slot_offset == lo
                slot_is_last = slot_offset + TRIPLET_SLOT_UNITS >= lo + ld
                if slot_is_first and slot_is_last:
                    active_tie = tie
                elif slot_is_first:
                    active_tie = "start" if tie in (None, "start") else "continue"
                elif slot_is_last:
                    active_tie = "stop" if tie in (None, "stop") else "continue"
                else:
                    active_tie = "continue"
                break
        slots.append((active_pitch, active_tie))

    bracket_positions = ["start", "middle", "stop"]
    for i, ((pitch, tie), bracket_pos) in enumerate(zip(slots, bracket_positions)):
        el = _make_triplet_element(TRIPLET_SLOT_UNITS, pitch, bracket_pos)
        if pitch is not None and tie is not None:
            el.tie = m21.tie.Tie(tie)
        offset_q = Fraction(
            quarter_offset_in_bar + i * TRIPLET_SLOT_UNITS, UNITS_PER_QUARTER
        )
        measure.insert(offset_q, el)


def _build_p2_quarter(
    events, measure: m21.stream.Measure, quarter_offset_in_bar: int
):
    """Render a power-of-2 mode quarter: notes/rests on 16th grid using
    standard durations (no triplets)."""
    cursor_in_q = 0
    for lo, ld, pitch, tie in events:
        if lo > cursor_in_q:
            measure.insert(
                Fraction(quarter_offset_in_bar + cursor_in_q, UNITS_PER_QUARTER),
                m21.note.Rest(
                    quarterLength=Fraction(lo - cursor_in_q, UNITS_PER_QUARTER)
                ),
            )
        pieces = _decompose_p2(ld)
        if not pieces:
            cursor_in_q = lo
            continue
        ties = _ties_for_pieces(len(pieces), tie)
        offset_in_q = lo
        for piece_units, piece_tie in zip(pieces, ties):
            n = m21.note.Note(
                pitch,
                quarterLength=Fraction(piece_units, UNITS_PER_QUARTER),
            )
            if piece_tie:
                n.tie = m21.tie.Tie(piece_tie)
            measure.insert(
                Fraction(quarter_offset_in_bar + offset_in_q, UNITS_PER_QUARTER), n
            )
            offset_in_q += piece_units
        cursor_in_q = offset_in_q
    if cursor_in_q < UNITS_PER_QUARTER:
        measure.insert(
            Fraction(quarter_offset_in_bar + cursor_in_q, UNITS_PER_QUARTER),
            m21.note.Rest(
                quarterLength=Fraction(
                    UNITS_PER_QUARTER - cursor_in_q, UNITS_PER_QUARTER
                )
            ),
        )


def _is_clean_single_duration(qL: float) -> bool:
    """True iff `qL` can be written as ONE standard non-tuplet note."""
    try:
        d = m21.duration.Duration(quarterLength=qL)
    except Exception:
        return False
    if d.type == "complex":
        return False
    if d.tuplets:
        return False
    if len(d.components) != 1:
        return False
    return True


def _merge_tied_chains_in_measure(measure: m21.stream.Measure) -> int:
    """Cosmetic pass: merge consecutive tied notes of the same pitch in one
    measure into a single note when the total duration is a clean standard
    duration. Triplet-bracketed notes are skipped. The merged note keeps
    the outer tie state if the chain is open (e.g. continues into the next
    measure). Returns count of merges."""
    elements = sorted(list(measure.notesAndRests), key=lambda e: float(e.offset))
    chains: list[list[m21.note.Note]] = []
    cur: list[m21.note.Note] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            chains.append(cur)
        cur = []

    for el in elements:
        if not isinstance(el, m21.note.Note) or el.duration.tuplets:
            flush()
            continue
        tie_type = el.tie.type if el.tie else None
        if not cur:
            if tie_type in ("start", "continue"):
                cur = [el]
            continue
        if el.pitch.midi != cur[-1].pitch.midi:
            flush()
            if tie_type in ("start", "continue"):
                cur = [el]
            continue
        cur.append(el)
        if tie_type == "stop":
            flush()
    flush()

    merged = 0
    for chain in chains:
        if len(chain) < 2:
            continue
        first = chain[0]
        last = chain[-1]
        first_tie = first.tie.type if first.tie else None
        last_tie = last.tie.type if last.tie else None
        first_offset = float(first.offset)
        pitch_midi = first.pitch.midi
        total = (
            float(last.offset)
            + float(last.duration.quarterLength)
            - first_offset
        )
        if not _is_clean_single_duration(total):
            continue
        if first_tie == "start" and last_tie == "stop":
            new_tie: str | None = None
        elif first_tie == "start" and last_tie in ("continue", None):
            new_tie = "start"
        elif first_tie == "continue" and last_tie == "stop":
            new_tie = "stop"
        elif first_tie == "continue" and last_tie in ("continue", None):
            new_tie = "continue"
        else:
            continue
        for el in chain:
            measure.remove(el)
        new_note = m21.note.Note(pitch_midi, quarterLength=total)
        if new_tie:
            new_note.tie = m21.tie.Tie(new_tie)
        measure.insert(first_offset, new_note)
        merged += 1
    return merged


def _merge_tied_chains(score: m21.stream.Score) -> int:
    total = 0
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            total += _merge_tied_chains_in_measure(measure)
    return total


def _empty_score() -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()
    part.append(m21.meter.TimeSignature("4/4"))
    measure = m21.stream.Measure(number=1)
    measure.append(m21.note.Rest(quarterLength=DEFAULT_BEATS_PER_BAR))
    part.append(measure)
    score.append(part)
    return score


class MidiToMusicxmlConverter:
    @staticmethod
    def to_melody_only(midi: pretty_midi.PrettyMIDI) -> m21.stream.Score:
        """Render PrettyMIDI to an m21 Score by building Part / Measure /
        Note objects directly. Each bar is split into 4 quarters, and each
        quarter renders in either triplet or power-of-2 mode based on note
        positions. Tied chains are merged into clean standard durations
        where possible. Source `midi` is not mutated."""
        notes = _collect_notes_in_units(midi)
        if not notes:
            return _empty_score()

        bpm, key = _read_metadata(midi)
        last_end_u = max(end_u for _, end_u, _ in notes)
        total_bars = math.ceil(last_end_u / UNITS_PER_BAR)

        bar_q_events: dict[
            tuple[int, int], list[tuple[int, int, int, str | None]]
        ] = {}
        for start_u, end_u, pitch in notes:
            for bar_idx, q_idx, lo, ld, tie in _split_into_quarter_fragments(
                start_u, end_u
            ):
                bar_q_events.setdefault((bar_idx, q_idx), []).append(
                    (lo, ld, pitch, tie)
                )

        score = m21.stream.Score()
        part = m21.stream.Part()

        for bar_idx in range(total_bars):
            measure = m21.stream.Measure(number=bar_idx + 1)
            if bar_idx == 0:
                # MusicXML carries tempo / time / key signature inside the
                # first measure; placing them on the Part-level ahead of
                # measures gets silently dropped by m21's writer.
                measure.insert(0.0, m21.tempo.MetronomeMark(number=bpm))
                measure.insert(0.0, m21.meter.TimeSignature("4/4"))
                if key is not None:
                    measure.insert(0.0, key)
            for q_idx in range(DEFAULT_BEATS_PER_BAR):
                events = sorted(bar_q_events.get((bar_idx, q_idx), []))
                quarter_offset_in_bar = q_idx * UNITS_PER_QUARTER
                if not events:
                    measure.insert(
                        Fraction(quarter_offset_in_bar, UNITS_PER_QUARTER),
                        m21.note.Rest(
                            quarterLength=Fraction(
                                UNITS_PER_QUARTER, UNITS_PER_QUARTER
                            )
                        ),
                    )
                    continue
                mode = _quarter_mode(events)
                events = _snap_events_to_mode(events, mode)
                if not events:
                    measure.insert(
                        Fraction(quarter_offset_in_bar, UNITS_PER_QUARTER),
                        m21.note.Rest(
                            quarterLength=Fraction(
                                UNITS_PER_QUARTER, UNITS_PER_QUARTER
                            )
                        ),
                    )
                    continue
                if mode == "triplet":
                    _build_triplet_quarter(events, measure, quarter_offset_in_bar)
                else:
                    _build_p2_quarter(events, measure, quarter_offset_in_bar)
            part.append(measure)

        score.append(part)
        _merge_tied_chains(score)
        return score

    @staticmethod
    def to_with_chords(
        midi: pretty_midi.PrettyMIDI,
        theme_chord_symbols: list[tuple[float, m21.harmony.ChordSymbol]],
        input_bars: int,
        output_bars: int,
        beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    ) -> m21.stream.Score:
        """Render to m21 Score and add the theme's chord symbols on top,
        tiled across all bars (theme + improvisation) cyclically.

        Each chord_symbol is deep-copied before insertion so cycles do not
        share mutable m21 state, and we avoid re-parsing the figure
        (m21 chokes on extended figures like "Eb7 add b9" during
        ChordSymbol(figure) round-trip).

        Cyclic tiling = the harmonic progression of the theme repeats over
        the generated section — same convention every wrapper uses
        internally (CMT cycles its chord tensor; MINGUS plays choruses;
        BebopNet generates over the head's chord progression).
        """
        score = MidiToMusicxmlConverter.to_melody_only(midi)
        if not theme_chord_symbols:
            return score

        cycle_length_q = float(input_bars * beats_per_bar)
        total_q = float((input_bars + output_bars) * beats_per_bar)
        num_cycles = math.ceil(total_q / cycle_length_q) if cycle_length_q > 0 else 0

        target_part = score.parts[0] if score.parts else score
        for cycle_idx in range(num_cycles):
            cycle_offset_q = cycle_idx * cycle_length_q
            for offset_q, chord_symbol in theme_chord_symbols:
                placed_offset = cycle_offset_q + offset_q
                if placed_offset >= total_q:
                    break
                _insert_chord_symbol_inside_measure(
                    target_part, placed_offset, copy.deepcopy(chord_symbol)
                )

        return score


def _insert_chord_symbol_inside_measure(
    part: m21.stream.Stream,
    placed_offset_q: float,
    chord_symbol: m21.harmony.ChordSymbol,
) -> None:
    """ChordSymbol elements only round-trip through MusicXML's <harmony>
    when they live inside a <measure>. Find the measure that spans
    placed_offset_q and insert the chord_symbol there at the local offset.

    If the part has no measures (e.g. a fresh m21 stream from midi parsing
    that wasn't `makeMeasures`-d), fall back to part-level insert.
    """
    measures = list(part.getElementsByClass(m21.stream.Measure))
    if not measures:
        part.insert(placed_offset_q, chord_symbol)
        return

    for measure in measures:
        measure_start = float(measure.offset)
        measure_length = float(measure.duration.quarterLength)
        if measure_start <= placed_offset_q < measure_start + measure_length:
            measure.insert(placed_offset_q - measure_start, chord_symbol)
            return

    # placed_offset_q overshoots the last measure — drop silently rather
    # than raise: caller's bar-math may produce one out-of-range entry
    # at boundaries.
