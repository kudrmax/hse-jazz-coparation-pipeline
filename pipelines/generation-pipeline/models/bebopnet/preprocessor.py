"""BebopnetPreprocessor — extends CommonPreprocessor with two extra steps:

1. Selective detie: untie tied chains whose accumulated quarterLength is
   not in the model's duration vocabulary (would otherwise crash BebopNet's
   tie-merge loop with KeyError on lookup).
2. Snap-argmin: replace any individual note's quarterLength with the
   nearest in-vocab value if the original is off-vocab (e.g. 5-tuplets).

See docs/superpowers/specs/2026-05-08-bebopnet-duration-snap-design.md.
"""
from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import music21 as m21

from models.base.preprocessor import CommonPreprocessor


class BebopnetPreprocessor(CommonPreprocessor):
    def __init__(self, vocab: frozenset[Fraction]) -> None:
        """Cache the duration vocabulary used by detie/snap.

        `vocab` is the set of in-vocab quarterLength fractions (drawn from
        the BebopNet converter's `bidict.keys()`). Pipeline-venv code
        obtains this set by running `_vocab_dump_runner.py` inside the
        bebopnet venv, since the converter pickle requires a bidict-shim
        only available in the bebopnet fork.
        """
        super().__init__()
        self._vocab: frozenset[Fraction] = vocab
        self._vocab_sorted: tuple[Fraction, ...] = tuple(sorted(vocab))

    def _iter_tied_chains(
        self, stream: m21.stream.Score
    ) -> list[list[m21.note.Note]]:
        """Group consecutive notes connected by ties into chains.

        A chain is a list of one or more Note objects where each note's
        tie attribute follows the pattern start → continue* → stop.
        Untied notes form singleton chains of length 1.

        Returns a list of chains, in stream order.
        """
        chains: list[list[m21.note.Note]] = []
        current: list[m21.note.Note] = []
        for note in stream.recurse().getElementsByClass(m21.note.Note):
            tie_type = note.tie.type if note.tie is not None else None
            if tie_type == "start":
                if current:
                    chains.append(current)
                current = [note]
            elif tie_type in ("continue", "stop"):
                if current:
                    current.append(note)
                else:
                    # Orphan tie-stop without a tie-start; treat as singleton.
                    chains.append([note])
            else:
                # Untied note: flush any open chain, then add as singleton.
                if current:
                    chains.append(current)
                    current = []
                chains.append([note])
        if current:
            chains.append(current)
        return chains

    def _detie_off_vocab_chains(self, stream: m21.stream.Score) -> bool:
        """For each tied chain whose accumulated quarterLength is not in
        self._vocab, set tie=None on every note in that chain.

        Returns True if any chain was detied, False if no changes.
        Singleton chains (untied notes) are skipped — they have no ties.
        """
        modified = False
        for chain in self._iter_tied_chains(stream):
            if len(chain) < 2:
                continue  # singleton, no ties to consider
            total = sum((Fraction(n.quarterLength) for n in chain), Fraction(0))
            total = total.limit_denominator(96)
            if total in self._vocab:
                continue  # legal tied chain, leave alone
            for note in chain:
                note.tie = None
            modified = True
        return modified

    def _snap_off_vocab_individual_notes(self, stream: m21.stream.Score) -> bool:
        """For each Note whose individual quarterLength is not in the cached
        vocab, replace its qL with the nearest in-vocab value.

        Tie-break for equidistant: prefer the smaller value (avoids extending
        notes past bar boundaries that might trigger further surprises).

        Returns True if any note was modified, False otherwise.
        """
        modified = False
        for note in stream.recurse().getElementsByClass(m21.note.Note):
            qL = Fraction(note.quarterLength).limit_denominator(96)
            if qL in self._vocab:
                continue
            # argmin by |d - qL|; on equidistance, sorted vocab gives the
            # smaller value first because min() is stable and we sort ascending.
            nearest = min(
                self._vocab_sorted,
                key=lambda d, qL=qL: abs(d - qL),
            )
            note.quarterLength = float(nearest)
            modified = True
        return modified

    def process(
        self,
        inp: Any,
        parsed_stream: m21.stream.Score,
    ) -> tuple[m21.stream.Score, Path]:
        """Run super (trim to input_bars), then selective detie + snap.
        If we modify the stream, write a new tmp musicxml and return its
        path; clean up super's tmp file if super produced one.
        """
        stream, path_after_super = super().process(inp, parsed_stream)

        detied = self._detie_off_vocab_chains(stream)
        snapped = self._snap_off_vocab_individual_notes(stream)

        if not (detied or snapped):
            return stream, path_after_super

        # We modified the stream; write a new tmp file.
        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False
        ) as f:
            our_tmp = Path(f.name)
        actual = Path(str(stream.write("musicxml", fp=str(our_tmp))))
        if actual != our_tmp:
            actual.replace(our_tmp)

        # If super produced its own tmp (path_after_super != original),
        # we must clean it up since BaseGenerator.generate only cleans the
        # FINAL returned path.
        if path_after_super != inp.get_musicxml_path():
            path_after_super.unlink(missing_ok=True)

        return stream, our_tmp
