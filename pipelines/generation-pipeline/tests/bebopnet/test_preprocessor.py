"""BebopnetPreprocessor inherits CommonPreprocessor (Z-style)."""
from models.base.preprocessor import CommonPreprocessor
from models.bebopnet.preprocessor import BebopnetPreprocessor


def test_bebopnet_preprocessor_is_subclass_of_common():
    assert issubclass(BebopnetPreprocessor, CommonPreprocessor)
