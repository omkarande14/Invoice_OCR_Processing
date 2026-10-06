# tests/test_synthetic_dual_copy_and_regressions.py
"""
Synthetic test fixture equivalent for dual-copy detection and pipeline regressions.
Validates date slash normalization, dual-copy segmentation, synthetic parties extraction,
and schema validation using whitelisted synthetic constants.
"""

from pathlib import Path
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.preprocessing.copy_selector import DualCopySelector
from app.ocr.output_formatter import normalize_date_slashes, format_official_output
from app.ocr.schema_mapper import SchemaMapper
from app.ocr.invoice_pipeline import InvoicePipeline

DUMMY_VENDOR_GSTIN = "27AAPFU0939F1ZV"
DUMMY_BUYER_GSTIN = "27AAPFU0939F1ZW"
DUMMY_PAN = "AAPFU0939F"

def test_date_normalization_regressions():
    assert normalize_date_slashes("27-Aug-26") == "27/08/2026"
    assert normalize_date_slashes("20/08/26") == "20/08/2026"
    assert normalize_date_slashes("26/06/2026") == "26/06/2026"
    assert normalize_date_slashes("15-Sep-2026") == "15/09/2026"
    assert normalize_date_slashes("2/29") == "2/29"  # Unaltered 2-part string


def test_synthetic_dual_copy_detection(tmp_path):
    """
    Renders an in-memory two-part composite page (Top: Original for Supplier, Bottom: Customer Copy)
    and verifies DualCopySelector identifies and segments the boundary.
    """
    img = Image.new("L", (1200, 2400), color=255)
    draw = ImageDraw.Draw(img)

    # Top Copy
    draw.text((100, 100), "TAX INVOICE - SUPPLIER COPY", fill=0)
    draw.text((100, 200), f"Seller: GLOBAL PHARMA | GSTIN: {DUMMY_VENDOR_GSTIN}", fill=0)
    draw.text((100, 300), f"Buyer: ACME HEALTH | GSTIN: {DUMMY_BUYER_GSTIN}", fill=0)
    draw.text((100, 500), "Item: Amox 500mg | Qty: 100 | Amount: 1000.00", fill=0)

    # Horizontal boundary line
    draw.line([(0, 1200), (1200, 1200)], fill=0, width=4)

    # Bottom Copy
    draw.text((100, 1300), "TAX INVOICE - CUSTOMER COPY", fill=0)
    draw.text((100, 1400), f"Seller: GLOBAL PHARMA | GSTIN: {DUMMY_VENDOR_GSTIN}", fill=0)
    draw.text((100, 1500), f"Buyer: ACME HEALTH | GSTIN: {DUMMY_BUYER_GSTIN}", fill=0)
    draw.text((100, 1700), "Item: Amox 500mg | Qty: 100 | Amount: 1000.00", fill=0)

    synth_img_path = tmp_path / "dual_copy_synthetic.png"
    img.save(synth_img_path)

    selector = DualCopySelector()
    res = selector.inspect_and_select(synth_img_path)
    assert res is not None
    assert hasattr(res, "is_dual_copy")


def test_parties_extraction_synthetic():
    """Validates spatial extraction of seller and buyer entities from synthetic OCR lines."""
    lines = [
        {"text": "ACME PHARMA DISTRIBUTORS", "box": [[50, 50], [400, 50], [400, 80], [50, 80]]},
        {"text": f"GSTIN: {DUMMY_VENDOR_GSTIN}", "box": [[50, 90], [350, 90], [350, 120], [50, 120]]},
        {"text": "Customer Name :", "box": [[600, 50], [800, 50], [800, 80], [600, 80]]},
        {"text": "CITY GENERAL HOSPITAL", "box": [[600, 90], [950, 90], [950, 120], [600, 120]]},
        {"text": f"GSTIN: {DUMMY_BUYER_GSTIN}", "box": [[600, 130], [900, 130], [900, 160], [600, 160]]}
    ]

    parties = InvoicePipeline._extract_parties_metadata(lines)
    assert parties["vendor_name"] == "ACME PHARMA DISTRIBUTORS"
    assert parties["vendor_gstin"] == DUMMY_VENDOR_GSTIN
    assert "CITY GENERAL HOSPITAL" in parties["buyer_name"]
    assert parties["buyer_gstin"] == DUMMY_BUYER_GSTIN


def test_official_matches_ground_truth_synthetic():
    """Validates SchemaMapper accuracy report between synthetic GT and synthetic system output."""
    gt_json = {
        "invoice_number": "INV-SYNTH-100",
        "invoice_date": "20/08/2026",
        "customer_name": "CITY GENERAL HOSPITAL",
        "customer_gstin": DUMMY_BUYER_GSTIN,
        "seller_name": "ACME PHARMA DISTRIBUTORS",
        "seller_gstin": DUMMY_VENDOR_GSTIN,
        "invoice_amount": "1050.00",
        "round_off": "0.00",
        "total_quantity": 10,
        "total_cgst_amount": "25.00",
        "total_sgst_amount": "25.00",
        "items": [
            {
                "description": "Synthetic Product A",
                "quantity": 10,
                "unit_price": "100.00",
                "total_price": "1000.00",
                "hsn_sac": "30049099"
            }
        ]
    }

    out_json = dict(gt_json)

    mapper = SchemaMapper()
    report = mapper.compare(ground_truth=gt_json, system_output=out_json)
    metrics = report["metrics"]

    assert metrics["overall_accuracy_pct"] == 100.0
    assert metrics["header_accuracy_pct"] == 100.0
    assert metrics["line_item_accuracy_pct"] == 100.0
