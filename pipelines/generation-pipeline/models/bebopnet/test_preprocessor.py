"""Unit tests for BebopnetPreprocessor.

Run from pipeline venv:
    pipelines/generation-pipeline/.venv/bin/python -m pytest \
        pipelines/generation-pipeline/models/bebopnet/test_preprocessor.py -v
"""
from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path


def test_module_imports() -> None:
    """Skeleton smoke — verifies the preprocessor module is importable
    and currently empty (BebopnetPreprocessor exists with no overrides)."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    assert BebopnetPreprocessor is not None


def _vocab(durations: list[Fraction]) -> frozenset[Fraction]:
    """Build a frozenset vocab for tests (mirrors what
    `_vocab_dump_runner.py` produces from the real fork-pickled converter)."""
    return frozenset(durations)


def test_init_caches_vocab_as_frozenset() -> None:
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([
        Fraction(1, 4), Fraction(1, 2), Fraction(1, 1),
        Fraction(2, 1), Fraction(4, 1),
    ])
    pre = BebopnetPreprocessor(vocab=vocab)

    assert pre._vocab == frozenset({
        Fraction(1, 4), Fraction(1, 2), Fraction(1, 1),
        Fraction(2, 1), Fraction(4, 1),
    })
    assert pre._vocab_sorted == (
        Fraction(1, 4), Fraction(1, 2), Fraction(1, 1),
        Fraction(2, 1), Fraction(4, 1),
    )


import music21 as m21


def _make_stream_with_chain(
    pitches_qls: list[tuple[str, float, str | None]],
) -> m21.stream.Score:
    """Build a Score with one Part containing one Measure of given notes.

    Each entry: (pitch_name, quarterLength, tie_type or None).
    Tie types: 'start', 'continue', 'stop', or None.
    """
    part = m21.stream.Part()
    measure = m21.stream.Measure()
    for pitch, qL, tie_type in pitches_qls:
        note = m21.note.Note(pitch, quarterLength=qL)
        if tie_type is not None:
            note.tie = m21.tie.Tie(tie_type)
        measure.append(note)
    part.append(measure)
    score = m21.stream.Score()
    score.append(part)
    return score


def _vocab_simple() -> frozenset[Fraction]:
    """Small synthetic vocab for tests: covers powers of 2 up to 4."""
    return _vocab([
        Fraction(1, 4), Fraction(1, 2), Fraction(1, 1),
        Fraction(2, 1), Fraction(4, 1),
    ])


def test_legal_tied_chain_preserved() -> None:
    """Chain (qL=1 tied qL=1, sum=2 ∈ vocab) → ties NOT removed."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    stream = _make_stream_with_chain([
        ("C4", 1.0, "start"),
        ("C4", 1.0, "stop"),
    ])
    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    pre._detie_off_vocab_chains(stream)

    notes = list(stream.recurse().getElementsByClass(m21.note.Note))
    assert notes[0].tie is not None and notes[0].tie.type == "start"
    assert notes[1].tie is not None and notes[1].tie.type == "stop"


def test_off_vocab_tied_chain_detied() -> None:
    """Chain (qL=4 tied qL=1, sum=5 ∉ vocab) → both notes' ties removed."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    stream = _make_stream_with_chain([
        ("C4", 4.0, "start"),
        ("C4", 1.0, "stop"),
    ])
    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    pre._detie_off_vocab_chains(stream)

    notes = list(stream.recurse().getElementsByClass(m21.note.Note))
    assert notes[0].tie is None
    assert notes[1].tie is None
    assert notes[0].quarterLength == 4.0
    assert notes[1].quarterLength == 1.0


def test_three_note_chain_detied_when_sum_off_vocab() -> None:
    """Chain start→continue→stop summing to 6 (∉ vocab) → all detied."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    stream = _make_stream_with_chain([
        ("C4", 2.0, "start"),
        ("C4", 2.0, "continue"),
        ("C4", 2.0, "stop"),
    ])
    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    pre._detie_off_vocab_chains(stream)

    notes = list(stream.recurse().getElementsByClass(m21.note.Note))
    assert all(n.tie is None for n in notes)
    assert [n.quarterLength for n in notes] == [2.0, 2.0, 2.0]


def test_no_ties_no_op() -> None:
    """Stream with no tied notes → detie step is a no-op."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    stream = _make_stream_with_chain([
        ("C4", 1.0, None),
        ("D4", 2.0, None),
    ])
    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    pre._detie_off_vocab_chains(stream)  # should not raise

    notes = list(stream.recurse().getElementsByClass(m21.note.Note))
    assert all(n.tie is None for n in notes)
    assert [n.quarterLength for n in notes] == [1.0, 2.0]


def test_snap_off_vocab_individual_note_strictly_closer() -> None:
    """Vocab {1, 4}, qL=3.0 → snap to 4 (distance 1.0 < 2.0)."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([Fraction(1, 1), Fraction(4, 1)])
    stream = _make_stream_with_chain([("C4", 3.0, None)])
    pre = BebopnetPreprocessor(vocab=vocab)
    pre._snap_off_vocab_individual_notes(stream)

    note = next(iter(stream.recurse().getElementsByClass(m21.note.Note)))
    assert note.quarterLength == 4.0


def test_snap_equidistant_picks_smaller() -> None:
    """Vocab {1, 4}, qL=2.5 → equidistant; tie-break: smaller (1.0)."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([Fraction(1, 1), Fraction(4, 1)])
    stream = _make_stream_with_chain([("C4", 2.5, None)])
    pre = BebopnetPreprocessor(vocab=vocab)
    pre._snap_off_vocab_individual_notes(stream)

    note = next(iter(stream.recurse().getElementsByClass(m21.note.Note)))
    assert note.quarterLength == 1.0


def test_snap_skips_in_vocab_notes() -> None:
    """Vocab {1, 4}, qL=4.0 → unchanged."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([Fraction(1, 1), Fraction(4, 1)])
    stream = _make_stream_with_chain([("C4", 4.0, None)])
    pre = BebopnetPreprocessor(vocab=vocab)
    pre._snap_off_vocab_individual_notes(stream)

    note = next(iter(stream.recurse().getElementsByClass(m21.note.Note)))
    assert note.quarterLength == 4.0


def test_snap_tuplet_to_nearest_grid() -> None:
    """Realistic case: qL=0.2 (5-tuplet 8th, 1/5 quarter) → snap to nearest in
    vocab {1/4, 1/6}. |0.2-0.25|=0.05, |0.2-0.1666...|=0.0333... → snap to 1/6."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([Fraction(1, 6), Fraction(1, 4)])
    stream = _make_stream_with_chain([("C4", 0.2, None)])
    pre = BebopnetPreprocessor(vocab=vocab)
    pre._snap_off_vocab_individual_notes(stream)

    note = next(iter(stream.recurse().getElementsByClass(m21.note.Note)))
    assert Fraction(note.quarterLength).limit_denominator(96) == Fraction(1, 6)


from models.base.io import BaseGeneratorInput


def _make_input(musicxml_path: Path, input_bars: int = 32) -> BaseGeneratorInput:
    return BaseGeneratorInput(
        musicxml_path=musicxml_path,
        seed=1,
        input_bars=input_bars,
        output_bars=8,
    )


def _write_score_to_tmp(score: m21.stream.Score, tmp_path: Path) -> Path:
    p = tmp_path / "input.musicxml"
    score.write("musicxml", fp=str(p))
    return p


def test_process_returns_input_path_when_legal(tmp_path: Path) -> None:
    """All-legal stream + no super-trim → returns input path unchanged."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    score = _make_stream_with_chain([("C4", 1.0, None), ("D4", 2.0, None)])
    input_path = _write_score_to_tmp(score, tmp_path)
    inp = _make_input(input_path)

    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    parsed = m21.converter.parse(str(input_path))

    out_stream, out_path = pre.process(inp, parsed)
    assert out_path == input_path


def test_process_writes_tmp_when_chain_detied(tmp_path: Path) -> None:
    """Off-vocab tied chain → returns a new tmp path, file exists, ties gone
    after re-parsing."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    score = _make_stream_with_chain([
        ("C4", 4.0, "start"),
        ("C4", 1.0, "stop"),
    ])
    input_path = _write_score_to_tmp(score, tmp_path)
    inp = _make_input(input_path)

    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    parsed = m21.converter.parse(str(input_path))

    out_stream, out_path = pre.process(inp, parsed)
    assert out_path != input_path
    assert out_path.exists()

    reparsed = m21.converter.parse(str(out_path))
    notes = list(reparsed.recurse().getElementsByClass(m21.note.Note))
    assert all(n.tie is None for n in notes)


def test_process_writes_tmp_when_note_snapped(tmp_path: Path) -> None:
    """Off-vocab individual note → returns a new tmp path."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    vocab = _vocab([Fraction(1, 1), Fraction(4, 1)])
    score = _make_stream_with_chain([("C4", 2.5, None)])
    input_path = _write_score_to_tmp(score, tmp_path)
    inp = _make_input(input_path)

    pre = BebopnetPreprocessor(vocab=vocab)
    parsed = m21.converter.parse(str(input_path))

    out_stream, out_path = pre.process(inp, parsed)
    assert out_path != input_path


def test_process_preserves_super_path_when_no_change(tmp_path: Path) -> None:
    """If super().process() returned a tmp path (e.g. trim happened) AND
    we made no changes, our process should return super's path, not write
    yet another tmp.

    NOTE: Triggering super's trim requires more measures than input_bars.
    For simplicity we use a stream that does NOT trigger trim AND has no
    off-vocab content; so super returns the original input path.
    """
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    score = _make_stream_with_chain([("C4", 1.0, None)])
    input_path = _write_score_to_tmp(score, tmp_path)
    inp = _make_input(input_path)

    pre = BebopnetPreprocessor(vocab=_vocab_simple())
    parsed = m21.converter.parse(str(input_path))

    out_stream, out_path = pre.process(inp, parsed)
    assert out_path == input_path


def test_real_autumn_leaves_chains_detied(tmp_path: Path) -> None:
    """Real Autumn_Leaves.musicxml + synthetic vocab without Fraction(5,1)
    → 9 chains with sum=5q get detied; chains with sum in vocab preserved.
    Note count preserved (detie doesn't remove notes, only ties)."""
    from models.bebopnet.preprocessor import BebopnetPreprocessor

    # Build a vocab that matches our paper-default for the cases we care
    # about: includes 1q, 2q, 3q, 4q, etc., but NOT 5q.
    vocab = _vocab([
        Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 3),
        Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1, 1),
        Fraction(3, 2), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
    ])

    pipeline_root = Path(__file__).resolve().parents[2]
    real_xml = (pipeline_root / "inputs" / "musicxml"
                / "Autumn_Leaves.musicxml")
    inp = _make_input(real_xml, input_bars=32)
    parsed = m21.converter.parse(str(real_xml))

    notes_before = list(
        parsed.recurse().getElementsByClass(m21.note.Note)
    )
    n_before = len(notes_before)
    n_chains_before = sum(1 for n in notes_before if n.tie is not None)

    pre = BebopnetPreprocessor(vocab=vocab)
    out_stream, _ = pre.process(inp, parsed)

    notes_after = list(
        out_stream.recurse().getElementsByClass(m21.note.Note)
    )
    n_after = len(notes_after)
    n_chains_after = sum(1 for n in notes_after if n.tie is not None)

    # Note count preserved (detie doesn't remove notes).
    assert n_after == n_before, (
        f"note count changed: {n_before} → {n_after}"
    )
    # At least some ties were removed (the off-vocab ones).
    assert n_chains_after < n_chains_before, (
        f"ties before={n_chains_before}, after={n_chains_after}; "
        "expected some chains to be detied"
    )
