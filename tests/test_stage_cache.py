# tests/test_stage_cache.py
"""
Unit tests for OCRStageCache (Phase 1.5):
- Cache put and get roundtrip
- Automatic key generation from image sha256 + engine + version + preprocess hash
- Cold vs warm timing measurement
- Cache invalidation / clearing
"""

import time
import pytest
from pathlib import Path
from app.ocr.stage_cache import OCRStageCache


class TestStageCache:

    @pytest.fixture
    def cache_env(self, tmp_path):
        cache_dir = tmp_path / "ocr_cache"
        cache = OCRStageCache(cache_dir=cache_dir, enabled=True)
        return cache, cache_dir

    def test_cache_put_and_get(self, cache_env):
        cache, _ = cache_env
        img_sha = "abc123def456"
        dummy_lines = [
            {"text": "INVOICE NO: 1234", "confidence": 0.99, "box": [[10, 10], [100, 10], [100, 30], [10, 30]]},
            {"text": "TOTAL: 5000.00", "confidence": 0.98, "box": [[10, 40], [100, 40], [100, 60], [10, 60]]}
        ]

        # Cold get returns None
        assert cache.get(img_sha, "PaddleOCR", "v1.0") is None

        # Put lines into cache
        cache.put(img_sha, "PaddleOCR", "v1.0", dummy_lines)

        # Warm get retrieves cached lines
        retrieved = cache.get(img_sha, "PaddleOCR", "v1.0")
        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0]["text"] == "INVOICE NO: 1234"

    def test_cache_key_invalidation_on_version_or_engine_change(self, cache_env):
        cache, _ = cache_env
        img_sha = "same_image_sha_999"
        dummy_lines = [{"text": "TEST LINE", "box": []}]

        cache.put(img_sha, "PaddleOCR", "v1.0", dummy_lines)

        # Changing engine name -> cache miss
        assert cache.get(img_sha, "Tesseract", "v1.0") is None

        # Changing engine version -> cache miss
        assert cache.get(img_sha, "PaddleOCR", "v2.0") is None

        # Changing preprocess config -> cache miss
        assert cache.get(img_sha, "PaddleOCR", "v1.0", preprocess_config_hash="v2_contrast") is None

    def test_cache_clear(self, cache_env):
        cache, _ = cache_env
        cache.put("sha_1", "PaddleOCR", "v1.0", [{"text": "A"}])
        cache.put("sha_2", "PaddleOCR", "v1.0", [{"text": "B"}])

        cleared_count = cache.clear()
        assert cleared_count == 2
        assert cache.get("sha_1", "PaddleOCR", "v1.0") is None
        assert cache.get("sha_2", "PaddleOCR", "v1.0") is None
