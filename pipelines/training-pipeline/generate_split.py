"""Generate wjazzd_split.json — single source of truth for train/eval/test
split across all three models (CMT, MINGUS, BebopNet).

Reproduces CMT's split logic exactly: random.seed(0), 80/10/10 by song_title.
Run once, commit the resulting JSON, all notebooks read it.

**Test-set policy:** test must be identical for cross-model comparison. If a
model cannot read certain files (e.g. MINGUS's authorial xml-converter fails
on 8 of 430 solos), those files are excluded from the canonical test set so
all three models compute their test metrics on the same subset. Train/eval
may differ between models — only test is fixed.

Excluded-from-test list is maintained per-model in EXCLUDED_FROM_TEST below.
Currently only MINGUS contributes; BebopNet failures will be added later.

TODO (cross-model SSoT adoption):
- MINGUS reads this JSON via `data_preprocessing.py --split-json`. ✅
- CMT reads this JSON via `preprocess.py --split-json` (cross-model SSoT
  branch). Existing pkl on disk is still on the legacy random.seed(0)
  branch (43-file test bucket); regeneration is optional (cosmetic). ✅
- BebopNet routes files via `wjazzd_split_prep.py` reading this JSON
  before `gather_data_from_xml.py`. ✅

TODO (test-set length stratification — open question):
- CMT cuts midi into fixed `num_bars`-tick windows; songs shorter than
  num_bars produce zero instances and silently drop from CMT's pkl.
- Out of 40 canonical test files, on CMT-16bars only 20 survive (98
  windows); on CMT-32bars likely 5-10. MINGUS and BebopNet work at the
  token level — all 40 always present.
- This breaks honest **per-window** cross-model comparison on
  CMT-32bars. Distributional cross-model metrics (FMD/OA/KL — what
  comparation-pipeline computes) are unaffected since they evaluate
  generated melodies, not per-token loss.
- If later we need honest per-window: re-roll split.json[test] to 40
  songs with ≥32 bars (max num_bars across CMT variants). Trade-off:
  loses representativeness — short forms (12-bar blues, 8-bar AABA,
  rhythm changes) drop out. Requires retraining all three models on
  the new split. Current decision: accept the limitation, document
  per-window CMT-32 gap in the thesis.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

EVAL_RATIO = 0.1
TEST_RATIO = 0.1

# Files that some model cannot process; excluded from canonical test set so
# all models compute test metrics on the same intersection. Surfaced here
# (not hidden in a separate config) so the lineage is reviewable in the diff.
EXCLUDED_FROM_TEST: List[str] = [
    # MINGUS: authorial wjazzDB_csv_to_xml.py fails (4-quarter Measure overflow
    # on dense 16th-note passages). See plan Task 2 self-review.
    "319_Miles_Davis_Miles_Runs_the_Voodoo_Down_Solo",
    "322_Miles_Davis_Orbits_Solo",
    "434_Wayne_Shorter_Orbits_Solo",
]


def cmt_base_split(midi_root: Path) -> Dict[str, List[str]]:
    """Generate the raw CMT split using preprocess.py:178-184 algorithm.

    Bit-exact reproduction of CMT's seed logic. This is what CMT's pkl files
    are locked to — must not change. Returns 80/10/10 over all eligible solos.
    """
    midi_root = Path(midi_root)
    if not midi_root.is_dir():
        raise FileNotFoundError(f"midi_root does not exist: {midi_root}")

    song_titles = sorted({p.name for p in midi_root.iterdir() if p.is_dir()})
    if not song_titles:
        raise ValueError(f"midi_root contains no song subdirectories: {midi_root}")

    n = len(song_titles)
    num_eval = int(n * EVAL_RATIO)
    num_test = int(n * TEST_RATIO)

    random.seed(0)
    eval_set = set(random.sample(sorted(set(song_titles)), num_eval))
    test_set = set(random.sample(sorted(set(song_titles) - eval_set), num_test))
    train_set = set(song_titles) - eval_set - test_set

    return {
        "train": sorted(train_set),
        "eval": sorted(eval_set),
        "test": sorted(test_set),
    }


def generate_split(
    midi_root: Path,
    excluded_from_test: List[str] = EXCLUDED_FROM_TEST,
) -> Dict[str, List[str]]:
    """Generate canonical split with cross-model-safe test set.

    Starts from the CMT base split, then removes ``excluded_from_test`` from
    the test bucket (these files are kept out — NOT moved to train, since CMT
    pkl is already locked to the original 344/43/43 partition). Train and eval
    are unchanged: each model handles missing files in those buckets locally.

    Raises:
        FileNotFoundError: midi_root does not exist.
        ValueError: midi_root is empty.
    """
    base = cmt_base_split(midi_root)
    excluded = set(excluded_from_test)
    return {
        "train": base["train"],
        "eval": base["eval"],
        "test": sorted(set(base["test"]) - excluded),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "models"
        / "CMT-pytorch"
        / "jazz"
        / "wjazzd"
        / "data"
        / "midi",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "wjazzd_split.json",
    )
    args = parser.parse_args()

    split = generate_split(args.midi_root)
    args.out.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {args.out}: train={len(split['train'])}, "
        f"eval={len(split['eval'])}, test={len(split['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
