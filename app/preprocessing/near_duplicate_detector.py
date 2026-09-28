# app/preprocessing/near_duplicate_detector.py

"""
CRITICAL ARCHITECTURAL PRINCIPLE:
---------------------------------
This module detects shared visual template/layout, NOT duplicate content,
and results must never be used to skip, merge, or exclude files from downstream processing.

Real-world verification on this dataset confirmed that different invoices
from the same vendor share identical or near-identical layouts (pHash distance 0-4)
while having completely different invoice numbers, line items, and GST amounts.

This module is maintained as an alias to `template_similarity_detector.py`.
"""

from app.preprocessing.template_similarity_detector import (
    compute_perceptual_hash,
    find_template_similar_images as find_near_duplicates,
    find_template_similar_images,
)

__all__ = [
    "compute_perceptual_hash",
    "find_near_duplicates",
    "find_template_similar_images",
]
