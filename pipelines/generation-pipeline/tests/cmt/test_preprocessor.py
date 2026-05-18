"""CmtPreprocessor inherits CommonPreprocessor (Z-style)."""
from models.base.preprocessor import CommonPreprocessor
from models.cmt.preprocessor import CmtPreprocessor


def test_cmt_preprocessor_is_subclass_of_common():
    assert issubclass(CmtPreprocessor, CommonPreprocessor)
