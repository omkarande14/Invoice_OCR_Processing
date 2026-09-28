# app/preprocessing/template_similarity_detector.py

"""
Template Similarity Detector (Milestone 3b)

CRITICAL ARCHITECTURAL PRINCIPLE:
---------------------------------
This module detects shared visual template/layout structures using perceptual
hashing (pHash), NOT document or content duplication.

Real-world evaluation on this dataset confirmed that different invoices issued
by the same vendor (e.g. INVOICE NO 556 vs Invoice no. 591, or DMH INV NO-4663
vs DMH INV NO-4938) share identical or near-identical visual layouts (Hamming
distance 0-4) while containing completely different line items, quantities,
and GST amounts.

Therefore:
1. Perceptual hashing is NOT a reliable duplicate-detection signal for this dataset.
2. These results must NEVER be used to skip, merge, or exclude files from downstream
   processing or OCR.
3. True "same invoice, different scan" deduplication must wait until post-OCR content
   extraction (comparing invoice numbers, dates, vendor GSTIN, and line items).
4. The layout groupings captured here remain valuable for downstream template-aware
   OCR and field extraction (Milestones 4-6).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
import imagehash


def compute_perceptual_hash(image_path: str, hash_size: int = 8) -> imagehash.ImageHash:
    """
    Computes a 64-bit perceptual hash (pHash) for an image using
    Discrete Cosine Transform (DCT).
    """
    with Image.open(image_path) as img:
        return imagehash.phash(img, hash_size=hash_size)


def find_template_similar_images(
    folder_path: str,
    target_files: Optional[List[str]] = None,
    threshold: int = 4,
    borderline_min: int = 5,
    borderline_max: int = 8,
) -> Dict[str, Any]:
    """
    Identifies images with shared visual layouts/templates among target files
    in folder_path using perceptual hashing (pHash) and Hamming distance.

    Args:
        folder_path: Directory containing the images.
        target_files: Optional explicit list of filenames to analyze.
                      If None, all PNG files in folder_path are scanned.
                      Typically, this should be the canonical unique images
                      identified from exact duplicate detection (Milestone 3a).
        threshold: Maximum Hamming distance to classify two images as
                   sharing a template layout (default: 4).
        borderline_min: Lower bound Hamming distance for borderline watchlist.
        borderline_max: Upper bound Hamming distance for borderline watchlist.

    Returns:
        A structured dictionary containing:
          - "threshold": Layout similarity cutoff (distance <= threshold)
          - "borderline_range": [min, max] for borderline layout watchlist
          - "total_images_analyzed": Number of images evaluated
          - "total_template_similarity_pairs": Count of pairs with distance <= threshold
          - "template_similarity_pairs": Detailed list of template-sharing pairs
          - "total_template_similarity_groups": Count of connected layout clusters
          - "template_similarity_groups": Clustered groups of images sharing templates
          - "total_borderline_pairs": Count of borderline layout pairs
          - "borderline_pairs": List of pairs with distance in [borderline_min, borderline_max]
          - "note": Warning that these indicate template similarity, not duplicate invoices.
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        return {
            "error": f"Folder not found: {folder_path}",
            "threshold": threshold,
            "borderline_range": [borderline_min, borderline_max],
            "total_images_analyzed": 0,
            "total_template_similarity_pairs": 0,
            "template_similarity_pairs": [],
            "total_template_similarity_groups": 0,
            "template_similarity_groups": [],
            "total_borderline_pairs": 0,
            "borderline_pairs": [],
            "note": "Detects shared visual template/layout, NOT duplicate content.",
        }

    # Determine files to process
    if target_files is not None:
        files_to_process = [f for f in sorted(target_files) if (folder / f).is_file()]
    else:
        files_to_process = sorted([item.name for item in folder.iterdir() if item.is_file() and item.suffix.lower() == ".png"])

    # Compute perceptual hash for each image
    file_hashes: Dict[str, imagehash.ImageHash] = {}
    for filename in files_to_process:
        filepath = folder / filename
        try:
            file_hashes[filename] = compute_perceptual_hash(str(filepath))
        except Exception:
            continue

    analyzed_filenames = list(file_hashes.keys())
    template_similarity_pairs = []
    borderline_pairs = []

    # Pairwise comparison
    total_files = len(analyzed_filenames)
    for i in range(total_files):
        for j in range(i + 1, total_files):
            file_1 = analyzed_filenames[i]
            file_2 = analyzed_filenames[j]
            h1 = file_hashes[file_1]
            h2 = file_hashes[file_2]

            dist = int(h1 - h2)
            similarity = round((1.0 - (dist / 64.0)) * 100, 2)

            pair_info = {
                "file_1": file_1,
                "file_2": file_2,
                "hamming_distance": dist,
                "similarity_percentage": similarity,
            }

            if dist <= threshold:
                template_similarity_pairs.append(pair_info)
            elif borderline_min <= dist <= borderline_max:
                borderline_pairs.append(pair_info)

    # Build connected clusters for template similarity groups
    adjacency: Dict[str, set] = {f: set() for f in analyzed_filenames}
    for pair in template_similarity_pairs:
        adjacency[pair["file_1"]].add(pair["file_2"])
        adjacency[pair["file_2"]].add(pair["file_1"])

    visited = set()
    template_similarity_groups = []
    for f in analyzed_filenames:
        if f not in visited and len(adjacency[f]) > 0:
            component = []
            queue = [f]
            visited.add(f)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in sorted(adjacency[curr]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component.sort()
            template_similarity_groups.append(component)

    return {
        "threshold": threshold,
        "borderline_range": [borderline_min, borderline_max],
        "total_images_analyzed": len(analyzed_filenames),
        "total_template_similarity_pairs": len(template_similarity_pairs),
        "template_similarity_pairs": template_similarity_pairs,
        "total_template_similarity_groups": len(template_similarity_groups),
        "template_similarity_groups": template_similarity_groups,
        "total_borderline_pairs": len(borderline_pairs),
        "borderline_pairs": borderline_pairs,
        "note": (
            "Detects shared visual template/layout, NOT duplicate content. "
            "Results must never be used to skip, merge, or exclude files from downstream processing."
        ),
    }
