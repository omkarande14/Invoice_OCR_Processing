# tests/test_synthetic_schema_mapper.py
"""
Synthetic test fixture equivalent for SchemaMapper.
Validates flat-to-flat and nested-to-flat schema mappings, scalar comparisons,
exempted typo matches, and line-item accuracy calculations using whitelisted synthetic constants.
"""

import pytest
from app.ocr.schema_mapper import SchemaMapper

DUMMY_VENDOR_GSTIN = "27AAPFU0939F1ZV"
DUMMY_BUYER_GSTIN = "27AAPFU0939F1ZW"
DUMMY_PAN = "AAPFU0939F"

def test_schema_mapper_comparison_with_exemptions():
    # Synthetic flat ground truth with known typo on item 2 item_code
    ground_truth = {
        "invoice_number": "INV-SYNTH-04531",
        "invoice_date": "26/06/2026",
        "customer_name": "ACME MEDICAL CARE",
        "customer_gstin": DUMMY_BUYER_GSTIN,
        "seller_name": "GLOBAL LIFECARE PRIVATE LIMITED",
        "seller_gstin": DUMMY_VENDOR_GSTIN,
        "invoice_amount": "252139.00",
        "round_off": "0.40",
        "total_cgst_amount": "6003.30",
        "total_sgst_amount": "6003.30",
        "items": [
            {
                "description": "ALBUCEL 20%/100ML",
                "Pack": "VIAL",
                "quantity": 60,
                "unit_price": "3700.00",
                "total_price": "222000.00",
                "hsn_sac": "30021290",
                "item_code": "AL-03-0279"
            },
            {
                "description": "TELIPSA 10ML/1MG INJ",
                "Pack": "AMP",
                "quantity": 120,
                "unit_price": "198.00",
                "total_price": "23760.00",
                "hsn_sac": "30049095",
                "item_code": "AL-02-3176" # Ground truth typo (actual is AL-02-3178)
            }
        ]
    }

    # System output (nested schema)
    system_output = {
        "invoice_metadata": {
            "invoice_number": "INV-SYNTH-04531",
            "invoice_date": "26-06-2026",
            "vendor": {
                "name": "GLOBAL LIFECARE PRIVATE LIMITED",
                "gstin": DUMMY_VENDOR_GSTIN,
                "pan": DUMMY_PAN,
                "bank_details": {
                    "bank_name": "SYNTHETIC BANK",
                    "account_number": "00000000000000",
                    "ifsc_code": "SYNB0000001"
                }
            },
            "buyer": {
                "name": "ACME MEDICAL CARE",
                "gstin": DUMMY_BUYER_GSTIN
            }
        },
        "line_items": [
            {
                "sr_no": 1,
                "item_code": "AL-03-0279",
                "description": "ALBUCEL 20%/100ML",
                "hsn_code": "30021290",
                "quantity": 60.0,
                "unit": "VIAL",
                "rate": 3700.0,
                "amount": 222000.0
            },
            {
                "sr_no": 2,
                "item_code": "AL-02-3178", # Real invoice text extracted by system
                "description": "TELIPSA 10ML/1MG INJ",
                "hsn_code": "30049095",
                "quantity": 120.0,
                "unit": "AMP",
                "rate": 198.0,
                "amount": 23760.0
            }
        ],
        "financial_summary": {
            "subtotal": 245760.0,
            "cgst": 6003.3,
            "sgst": 6003.3,
            "roundoff": 0.4,
            "grand_total": 252139.0
        }
    }

    mapper = SchemaMapper()
    report = mapper.compare(ground_truth, system_output)

    # All top-level scalar fields match
    for s in report["scalar_fields"]:
        assert s["match"] is True, f"Field mismatch: {s['field']}"

    # Item 1 matches perfectly
    item1_fields = {f["field"]: f for f in report["line_items"][0]["fields"]}
    assert item1_fields["item_code"]["match"] is True
    assert item1_fields["item_code"]["status"] == "MATCH"

    # Item 2 item_code is recognized as an exempted typo
    item2_fields = {f["field"]: f for f in report["line_items"][1]["fields"]}
    assert item2_fields["item_code"]["match"] is True
    assert item2_fields["item_code"]["is_exempted"] is True
    assert item2_fields["item_code"]["status"] == "EXEMPTED_TYPO_MATCH"

    # Overall accuracy is 100%
    assert report["metrics"]["overall_accuracy_pct"] == 100.0


def test_schema_mapper_direct_flat_comparison():
    """Tests direct flat-to-flat comparison without any nested translation."""
    ground_truth = {
        "invoice_id": "INV-SYNTH-04531",
        "invoice_number": "INV-SYNTH-04531",
        "invoice_date": "26/06/2026",
        "due_date": "26/06/2026",
        "customer_name": "ACME MEDICAL CARE",
        "customer_gstin": DUMMY_BUYER_GSTIN,
        "seller_name": "GLOBAL LIFECARE PRIVATE LIMITED",
        "seller_gstin": DUMMY_VENDOR_GSTIN,
        "currency_code": "INR",
        "PO_number": "PO/SYNTH/2026-27/426",
        "DC_date": None,
        "DC_number": None,
        "invoice_amount": "252139.00",
        "round_off": "0.40",
        "total_gst_rate": 5,
        "total_quantity": 180,
        "total_cgst_rate": 2.5,
        "total_cgst_amount": "6003.30",
        "total_sgst_rate": 2.5,
        "total_sgst_amount": "6003.30",
        "total_igst_rate": None,
        "total_igst_amount": "0.00",
        "total_gst_amount": "12006.60",
        "items": [
            {
                "description": "ALBUCEL 20%/100ML",
                "Pack": "VIAL",
                "Batch": "97010700",
                "quantity": 60,
                "free_item_yn": "0",
                "unit_price": "3700.00",
                "total_price": "222000.00",
                "reference_number": None,
                "hsn_sac": "30021290",
                "item_code": "AL-03-0279",
                "expiry_date": "28/02/2029",
                "Discount": "2.00",
                "Discount_type": None,
                "Value": None,
                "Gst%": 5,
                "MRP": "7986.00",
                "cgst_rate": 2.5,
                "cgst_amount": "5439.00",
                "sgst_rate": 2.5,
                "sgst_amount": "5439.00",
                "igst_rate": None,
                "igst_amount": None,
                "GST_AMT": "10878.00",
                "taxable_value": None
            },
            {
                "description": "TELIPSA 10ML/1MG INJ",
                "Pack": "AMP",
                "Batch": "A412507",
                "quantity": 120,
                "free_item_yn": "0",
                "unit_price": "198.00",
                "total_price": "23760.00",
                "reference_number": None,
                "hsn_sac": "30049095",
                "item_code": "AL-02-3176",
                "expiry_date": "30/11/2027",
                "Discount": "5.00",
                "Discount_type": None,
                "Value": None,
                "Gst%": 5,
                "MRP": "3289.69",
                "cgst_rate": 2.5,
                "cgst_amount": "564.30",
                "sgst_rate": 2.5,
                "sgst_amount": "564.30",
                "igst_rate": None,
                "igst_amount": None,
                "GST_AMT": "1128.60",
                "taxable_value": None
            }
        ]
    }

    # System output in identical flat schema (with AL-02-3178 extracted)
    system_output = dict(ground_truth)
    system_output["items"] = [
        dict(ground_truth["items"][0]),
        dict(ground_truth["items"][1])
    ]
    system_output["items"][1]["item_code"] = "AL-02-3178"

    mapper = SchemaMapper()
    report = mapper.compare(ground_truth, system_output)

    # All header fields match
    for h in report["header_fields"]:
        assert h["match"] is True, f"Header mismatch: {h['field']} exp={h['expected']} act={h['extracted']}"

    # All item fields match (including exempted typo)
    for it in report["line_items"]:
        for f in it["fields"]:
            assert f["match"] is True, f"Item field mismatch: {f['field']} exp={f['expected']} act={f['extracted']}"

    assert report["metrics"]["overall_accuracy_pct"] == 100.0
    assert report["metrics"]["overall_matches"] == "71/71"
