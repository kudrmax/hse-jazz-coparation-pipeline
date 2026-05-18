"""MingusPreprocessor inherits CommonPreprocessor (Z-style)."""
from models.base.preprocessor import CommonPreprocessor
from models.mingus.preprocessor import MingusPreprocessor


def test_mingus_preprocessor_is_subclass_of_common():
    assert issubclass(MingusPreprocessor, CommonPreprocessor)
