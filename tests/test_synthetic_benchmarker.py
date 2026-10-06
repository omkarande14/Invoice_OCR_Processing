# tests/test_synthetic_benchmarker.py
"""
Synthetic fixture tests for DualEngineBenchmarker.
Tests key entity extraction and consensus calculation on synthetic generated images.
Uses whitelisted synthetic tokens only.
"""

import pytest
from pathlib import Path
from PIL import Image, ImageDraw
from app.ocr.dual_engine_benchmarker import DualEngineBenchmarker

def test_synthetic_extract_key_entities():
    synthetic_text = """
    TAX INVOICE
    SYNTHETIC SUPPLIERS PVT LTD
    GSTIN No.: 27AAPFU0939F1ZV
    Invoice No. SYNTH-9876
    Invoice Date: 26/08/2026
    Total Amount: 14,520.50
    """
    entities = DualEngineBenchmarker.extract_key_entities(synthetic_text)
    assert "27AAPFU0939F1ZV" in entities["gstin"]
    assert any("SYNTH-9876" in inv for inv in entities["invoice_numbers"])
    assert any("26/08/2026" in d for d in entities["dates"])
    assert "14520.50" in entities["totals"]

def test_synthetic_consensus_scoring_logic(tmp_path):
    dummy_img_path = tmp_path / "synthetic_test_invoice.png"
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Invoice No: SYNTH-999\nGSTIN: 27AAPFU0939F1ZV\nTotal: 500.00", fill="black")
    img.save(dummy_img_path)

    benchmarker = DualEngineBenchmarker(use_gpu=False)
    comp = benchmarker.compare_image(dummy_img_path)

    assert "paddle_metrics" in comp
    assert "tesseract_metrics" in comp
    assert "consensus" in comp
    assert 0.0 <= comp["consensus_score"] <= 100.0
