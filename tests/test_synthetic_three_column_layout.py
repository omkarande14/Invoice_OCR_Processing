# tests/test_synthetic_three_column_layout.py
"""
Synthetic test fixture equivalent for three-column spatial layout extraction.
Validates spatial decomposition into vendor (left), metadata (center),
and buyer (right) using in-memory synthetic bounding boxes and whitelisted constants.
"""

import pytest
from app.ocr.invoice_pipeline import InvoicePipeline

DUMMY_VENDOR_GSTIN = "27AAPFU0939F1ZV"
DUMMY_BUYER_GSTIN = "27AAPFU0939F1ZW"
DUMMY_PAN = "AAPFU0939F"

def test_three_column_spatial_party_extraction():
    # Synthetic bounding boxes modeling the 3-column layout:
    # Left column (x: 50-500): Seller Synthetic Lifecare + GSTIN
    # Middle column (x: 700-1100): Invoice details (Invoice No, Date)
    # Right column (x: 1600-2200): Customer Synthetic Health + GSTIN
    elements = [
        # Seller (Left)
        {"text": "SYNTHETIC LIFECARE PRIVATE LIMITED", "box": [[60, 100], [500, 100], [500, 130], [60, 130]]},
        {"text": f"GSTIN : {DUMMY_VENDOR_GSTIN}", "box": [[60, 300], [400, 300], [400, 330], [60, 330]]},
        {"text": "Account No. : 00000000000000", "box": [[60, 400], [450, 400], [450, 430], [60, 430]]},
        {"text": "Bank Name: SYNTHETIC BANK", "box": [[60, 440], [350, 440], [350, 470], [60, 470]]},
        {"text": "IFSC : SYNB0000001", "box": [[60, 480], [300, 480], [300, 510], [60, 510]]},

        # Center (Invoice Details)
        {"text": "TAX INVOICE", "box": [[800, 50], [1000, 50], [1000, 80], [800, 80]]},
        {"text": "Invoice No.", "box": [[700, 120], [830, 120], [830, 150], [700, 150]]},
        {"text": "SYNTH/B/04531", "box": [[940, 120], [1100, 120], [1100, 150], [940, 150]]},
        {"text": "Invoice Date", "box": [[700, 160], [830, 160], [830, 190], [700, 190]]},
        {"text": "26/06/2026", "box": [[940, 160], [1050, 160], [1050, 190], [940, 190]]},

        # Right (Customer / Buyer)
        {"text": "Customer Name :", "box": [[1600, 100], [1800, 100], [1800, 130], [1600, 130]]},
        {"text": "SYNTHETIC MEDICAL STORES", "box": [[1600, 140], [2100, 140], [2100, 170], [1600, 170]]},
        {"text": f"GSTIN : {DUMMY_BUYER_GSTIN}", "box": [[1600, 250], [1950, 250], [1950, 280], [1600, 280]]}
    ]

    parties = InvoicePipeline._extract_parties_metadata(elements)
    assert parties["vendor_name"] == "SYNTHETIC LIFECARE PRIVATE LIMITED"
    assert parties["vendor_gstin"] == DUMMY_VENDOR_GSTIN
    assert parties["vendor_pan"] == DUMMY_PAN
    assert parties["buyer_name"] == "SYNTHETIC MEDICAL STORES"
    assert parties["buyer_gstin"] == DUMMY_BUYER_GSTIN
    assert parties["bank_details"]["account_number"] == "00000000000000"
    assert parties["bank_details"]["bank_name"] == "SYNTHETIC BANK"
    assert parties["bank_details"]["ifsc_code"] == "SYNB0000001"

    inv_num = InvoicePipeline._extract_invoice_number(elements)
    assert inv_num == "SYNTH/B/04531"

    inv_date = InvoicePipeline._extract_invoice_date(elements)
    assert inv_date == "26/06/2026"
