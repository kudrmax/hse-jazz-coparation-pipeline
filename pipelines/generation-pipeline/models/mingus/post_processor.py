"""MingusPostProcessor — currently no Mingus-specific post-processing.

Inherits common key_signature meta-event injection from
CommonPostProcessor. Kept as a separate class so that future
Mingus-only steps land here without disturbing the other wrappers.
"""
from __future__ import annotations

from models.base.post_processor import CommonPostProcessor


class MingusPostProcessor(CommonPostProcessor):
    pass
