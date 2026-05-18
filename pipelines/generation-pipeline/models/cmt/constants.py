"""Named constants for the CMT wrapper.

Two groups:
- "From CMT format" — values fixed by the upstream code/contract (do not change).
- "Music-theoretic / our choice" — values we pick that reflect domain assumptions.

PITCH_HOLD_TOKEN and PITCH_REST_TOKEN are NOT here: they depend on num_pitch
read from yaml at construction time and live as instance attributes on
GeneratorCmt.
"""
RHYTHM_REST = 0
RHYTHM_HOLD = 1
RHYTHM_ONSET = 2

CHORD_PITCH_CLASSES = 12

MELODY_INSTRUMENT_INDEX = 0

BEATS_PER_BAR = 4

TARGET_KEY_MAJOR = "C"
TARGET_KEY_MINOR = "A"
