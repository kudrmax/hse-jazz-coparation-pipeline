"""CmtPreprocessor — extends CommonPreprocessor.

Currently a thin subclass: the only preprocessing the pipeline applies
to CMT inputs is the common trim-to-input_bars step. CMT-specific
shaping (analyze_key + transpose to C/Am) still lives inside
GeneratorCmt._generate_impl; moving it here is left as a future
refactor (TODO in generator.py).
"""
from __future__ import annotations

from models.base.preprocessor import CommonPreprocessor


class CmtPreprocessor(CommonPreprocessor):
    pass
