"""
Shared utilities for the essay module.

Functions shared across multiple files to avoid copy-paste duplication.
"""

from __future__ import annotations

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on period, exclamation, and question mark."""
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


ARTICLE_RATIO_EXPECTED = 0.065
