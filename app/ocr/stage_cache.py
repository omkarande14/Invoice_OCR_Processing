# app/ocr/stage_cache.py
"""
Persistent Stage Cache for Raw OCR Engine Outputs (Phase 1.5).

Caches raw OCR line outputs keyed by:
  (image_sha256, engine_name, engine_version, preprocess_config_hash)
Automatically invalidates when any component changes.
Enables downstream stages (extraction, table parsing, self-healing, verification)
to re-run in milliseconds rather than ~25s per invoice.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class OCRStageCache:
    """Persistent on-disk cache for raw OCR output tokens and bounding boxes."""

    DEFAULT_CACHE_DIR = Path("data_private/stage_cache")

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_image_sha256(image_path_or_bytes: Any) -> str:
        """Computes SHA-256 of image bytes."""
        if isinstance(image_path_or_bytes, (str, Path)):
            p = Path(image_path_or_bytes)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            return hashlib.sha256(p.read_bytes()).hexdigest()
        elif isinstance(image_path_or_bytes, bytes):
            return hashlib.sha256(image_path_or_bytes).hexdigest()
        elif hasattr(image_path_or_bytes, "tobytes"):  # numpy array
            return hashlib.sha256(image_path_or_bytes.tobytes()).hexdigest()
        else:
            return hashlib.sha256(str(image_path_or_bytes).encode("utf-8")).hexdigest()

    def get_cache_key(
        self,
        image_sha256: str,
        engine_name: str,
        engine_version: str,
        preprocess_config_hash: str
    ) -> str:
        payload = f"{image_sha256}:{engine_name}:{engine_version}:{preprocess_config_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        image_sha256: str,
        engine_name: str,
        engine_version: str,
        preprocess_config_hash: str = "v1_default"
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieves cached OCR lines if present."""
        if not self.enabled:
            return None

        key = self.get_cache_key(image_sha256, engine_name, engine_version, preprocess_config_hash)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("ocr_lines")
        except Exception:
            return None

    def put(
        self,
        image_sha256: str,
        engine_name: str,
        engine_version: str,
        ocr_lines: List[Dict[str, Any]],
        preprocess_config_hash: str = "v1_default"
    ) -> None:
        """Stores raw OCR lines into persistent cache."""
        if not self.enabled:
            return

        key = self.get_cache_key(image_sha256, engine_name, engine_version, preprocess_config_hash)
        cache_file = self.cache_dir / f"{key}.json"
        payload = {
            "key": key,
            "image_sha256": image_sha256,
            "engine_name": engine_name,
            "engine_version": engine_version,
            "preprocess_config_hash": preprocess_config_hash,
            "line_count": len(ocr_lines),
            "ocr_lines": ocr_lines
        }
        try:
            tmp_file = self.cache_dir / f"{key}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            tmp_file.replace(cache_file)
        except Exception:
            pass

    def clear(self) -> int:
        """Clears all entries in the cache directory."""
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count
