"""BebopnetPostProcessor — currently no Bebopnet-specific post-processing.

Inherits common key_signature meta-event injection from
CommonPostProcessor. Kept as a separate class so that future
Bebopnet-only steps land here without disturbing the other wrappers.
"""
from __future__ import annotations

from models.base.post_processor import CommonPostProcessor


class BebopnetPostProcessor(CommonPostProcessor):
    pass
