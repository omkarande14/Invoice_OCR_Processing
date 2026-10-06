# tests/test_synthetic_schema_conformance.py
"""
Synthetic test fixture for official output schema conformance.
Verifies all required top-level keys, item keys, and data types on synthetic mock outputs.
Uses whitelisted synthetic tokens only.
"""

import pytest
from app.ocr.output_formatter import format_official_output

REQUIRED_HEADER_KEYS = [
    "invoice_id", "invoice_number", "invoice_date", "due_date",
    "customer_name", "customer_gstin", "seller_name", "seller_gstin",
    "currency_code", "PO_number", "DC_date", "DC_number",
    "invoice_amount", "round_off", "total_gst_rate", "total_quantity",
    "total_cgst_rate", "total_cgst_amount", "total_sgst_rate", "total_sgst_amount",
    "total_igst_rate", "total_igst_amount", "total_gst_amount", "items"
]

REQUIRED_ITEM_KEYS = [
    "description", "Pack", "Batch", "quantity", "free_quantity", "total_quantity",
    "free_item_yn", "unit_price", "total_price", "reference_number", "hsn_sac",
    "item_code", "expiry_date", "Discount", "Discount_type", "Value", "Gst%",
    "MRP", "cgst_rate", "cgst_amount", "sgst_rate", "sgst_amount", "igst_rate",
    "igst_amount", "GST_AMT", "taxable_value"
]

def test_synthetic_schema_structure_and_keys():
    synthetic_internal_data = {
        "file_name": "synthetic_inv_001.png",
        "invoice_metadata": {
            "invoice_number": "SYNTH-001",
            "invoice_date": "20/08/2026",
            "due_date": "10/09/2026",
            "currency_code": "INR",
            "PO_number": "PO-SYNTH-99",
            "vendor": {
                "name": "SYNTHETIC SUPPLIERS PVT LTD",
                "gstin": "27AAPFU0939F1ZV"
            },
            "buyer": {
                "name": "ACME HEALTHCARE CORP",
                "gstin": "27AAPFU0939F1ZW"
            }
        },
        "line_items": [
            {
                "description": "SYNTHETIC PHARMA SOLUTION 100ML",
                "unit": "100 ML",
                "batch": "SYNTH01",
                "qty": 10,
                "free_qty": 0,
                "rate": 100.0,
                "hsn_code": "30049099",
                "discount_pct": 5.0,
                "gst_pct": 5,
                "cgst_rate": 2.5,
                "cgst_amount": 23.75,
                "sgst_rate": 2.5,
                "sgst_amount": 23.75,
                "gst_amt": 47.50,
                "taxable_value": 950.00,
                "mrp": 150.00
            }
        ],
        "financial_summary": {
            "subtotal": 950.00,
            "cgst": 23.75,
            "sgst": 23.75,
            "igst": 0.0,
            "roundoff": 0.50,
            "grand_total": 998.00
        }
    }

    out = format_official_output(synthetic_internal_data)

    # 1. Check all required header keys exist
    for k in REQUIRED_HEADER_KEYS:
        assert k in out, f"Missing required header key: {k}"

    # 2. Check all required item keys exist
    assert len(out["items"]) == 1
    item = out["items"][0]
    for k in REQUIRED_ITEM_KEYS:
        assert k in item, f"Missing required item key: {k}"

    # 3. Check specific formatted types
    assert out["customer_gstin"] == "27AAPFU0939F1ZW"
    assert out["seller_gstin"] == "27AAPFU0939F1ZV"
    assert out["invoice_amount"] == "998.00"
    assert item["total_price"] == "950.00"
