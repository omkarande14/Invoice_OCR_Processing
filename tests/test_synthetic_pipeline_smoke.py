# tests/test_synthetic_pipeline_smoke.py
"""
Synthetic fixture smoke tests for InvoiceVerifier and InvoicePipeline.
Verifies GSTIN validation, arithmetic balance, and end-to-end extraction on synthetic image.
Uses whitelisted synthetic tokens only.
"""

import pytest
from PIL import Image, ImageDraw
from app.ocr.verifier import InvoiceVerifier
from app.ocr.invoice_pipeline import InvoicePipeline

def test_synthetic_gstin_validation():
    # Valid synthetic whitelisted GSTIN
    valid_res = InvoiceVerifier.validate_gstin("27AAPFU0939F1ZV")
    assert valid_res["valid"] is True
    assert valid_res["state_code"] == "27"

    # Invalid state code 99
    inv_state = InvoiceVerifier.validate_gstin("99AAPFU0939F1ZV")
    assert inv_state["valid"] is False

    # Invalid length
    inv_len = InvoiceVerifier.validate_gstin("27AAPFU0939")
    assert inv_len["valid"] is False

    # Non-string input
    assert InvoiceVerifier.validate_gstin(None)["valid"] is False

def test_synthetic_arithmetic_verification():
    line_items = [
        {"quantity": 10.0, "rate": 100.0, "amount": 1000.0},
        {"quantity": 2.0, "rate": 500.0, "amount": 1000.0}
    ]
    audit = InvoiceVerifier.verify_arithmetic(
        line_items=line_items,
        extracted_subtotal=2000.0,
        cgst=100.0,
        sgst=100.0,
        roundoff=0.0,
        extracted_grand_total=2200.0
    )
    assert audit["overall_status"] == "MATHEMATICALLY_VERIFIED"
    assert audit["all_items_verified"] is True
    assert audit["subtotal_verified"] is True
    assert audit["grand_total_verified"] is True
    assert audit["grand_total_source"] == "ocr_extracted"
    assert audit["reconciled_grand_total"] == 2200.0

def test_synthetic_pipeline_smoke_e2e(tmp_path):
    img_path = tmp_path / "synthetic_smoke_inv.png"
    img = Image.new("RGB", (1000, 600), color="white")
    draw = ImageDraw.Draw(img)

    text_lines = [
        "TAX INVOICE",
        "SYNTHETIC MEDICAL PVT LTD",
        "GSTIN: 27AAPFU0939F1ZV",
        "Invoice No: SYNTH-7777",
        "Date: 15/08/2026",
        "Billed To: ACME HOSPITAL",
        "Buyer GSTIN: 27AAPFU0939F1ZW",
        "Description   Qty   Rate   Amount",
        "BANDAGE PACK   10   50.00  500.00",
        "Subtotal: 500.00",
        "CGST (2.5%): 12.50",
        "SGST (2.5%): 12.50",
        "Total: 525.00"
    ]

    y = 30
    for line in text_lines:
        draw.text((50, y), line, fill="black")
        y += 35
    img.save(img_path)

    pipeline = InvoicePipeline(use_gpu=False)
    res = pipeline.process_invoice(str(img_path))

    assert "invoice_number" in res
    assert "seller_gstin" in res
    assert "invoice_amount" in res
    assert "items" in res
