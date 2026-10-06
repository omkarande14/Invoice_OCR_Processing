# tests/test_synthetic_vision_fallback.py
"""
Synthetic fixture unit tests for Multimodal Vision-AI Fallback Engine.
Tests availability, non-triggering on complete inputs, and mock patching.
Uses whitelisted synthetic tokens exclusively.
"""

from unittest.mock import MagicMock
from app.ocr.vision_fallback import GeminiVisionFallback

def test_synthetic_vision_fallback_without_api_key():
    fb = GeminiVisionFallback(api_key=None)
    assert fb.is_available() is False
    assert fb.extract_from_image("synthetic_dummy.png") is None

def test_synthetic_patch_missing_fields_noop_when_complete():
    fb = GeminiVisionFallback(api_key=None)
    complete_local_output = {
        "invoice_number": "SYNTH-101",
        "invoice_date": "01/08/2026",
        "due_date": "15/08/2026",
        "customer_name": "ACME HEALTHCARE CORP",
        "customer_gstin": "27AAPFU0939F1ZW",
        "seller_name": "SYNTHETIC SUPPLIERS PVT LTD",
        "seller_gstin": "27AAPFU0939F1ZV",
        "invoice_amount": "5000.00",
        "items": [{"description": "Item 1", "quantity": 10}]
    }

    patched = fb.patch_missing_fields(complete_local_output, "synthetic_dummy.png")
    assert patched == complete_local_output

def test_synthetic_patch_missing_fields_with_mocked_gemini():
    fb = GeminiVisionFallback(api_key="mock_key")
    fb._client = MagicMock()

    gemini_data = {
        "due_date": "20/08/2026",
        "customer_gstin": "27AAPFU0939F1ZW",
        "invoice_amount": "12500.00",
        "items": [
            {"description": "Synthetic Product A", "quantity": 100},
            {"description": "Synthetic Product B", "quantity": 50}
        ]
    }
    fb.extract_from_image = MagicMock(return_value=gemini_data)

    local_incomplete = {
        "invoice_number": "SYNTH-202",
        "invoice_date": "01/08/2026",
        "due_date": None,
        "customer_name": "ACME HEALTHCARE CORP",
        "customer_gstin": None,
        "seller_name": "SYNTHETIC SUPPLIERS PVT LTD",
        "seller_gstin": "27AAPFU0939F1ZV",
        "invoice_amount": None,
        "items": []
    }

    patched = fb.patch_missing_fields(local_incomplete, "synthetic_dummy.png")
    assert patched["due_date"] == "20/08/2026"
    assert patched["customer_gstin"] == "27AAPFU0939F1ZW"
    assert patched["invoice_amount"] == "12500.00"
    assert len(patched["items"]) == 2
