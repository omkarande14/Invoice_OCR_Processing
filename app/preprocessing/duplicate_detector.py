# app/preprocessing/duplicate_detector.py

from pathlib import Path
import hashlib
from PIL import Image


def compute_image_hash(image_path: str) -> str:
    """
    Returns a SHA-256 hash of the image's actual decoded pixel data.
    Two images with identical visual content produce the same hash,
    even if their PNG file bytes differ slightly (e.g. compression settings).
    """
    with Image.open(image_path) as img:
        pixel_data = img.tobytes()
    return hashlib.sha256(pixel_data).hexdigest()


def find_exact_duplicates(folder_path: str) -> dict:
    """
    Scans all PNG images in folder_path and groups exact visual duplicates.

    Returns:
      {
        "total_images": 79,
        "unique_images": 68,
        "total_duplicate_groups": 5,
        "redundant_copies": 11,
        "duplicate_groups": [
            ["fileA_page_1.png", "fileB_page_1.png"],
            ...
        ]
      }
    """
    folder = Path(folder_path)
    hash_to_files = {}

    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() == ".png":
            image_hash = compute_image_hash(str(item))
            hash_to_files.setdefault(image_hash, []).append(item.name)

    duplicate_groups = [files for files in hash_to_files.values() if len(files) > 1]
    redundant_copies = sum(len(group) - 1 for group in duplicate_groups)

    return {
        "total_images": sum(len(v) for v in hash_to_files.values()),
        "unique_images": len(hash_to_files),
        "total_duplicate_groups": len(duplicate_groups),
        "redundant_copies": redundant_copies,
        "duplicate_groups": duplicate_groups,
    }