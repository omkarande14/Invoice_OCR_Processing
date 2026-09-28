# tests/test_reconciliation_bug.py
"""
Test reproducing the reconciliation bug and verifying the fix from fixed numbers:
1. Invoice 4531 (Vendor A): Discrepancy explanation (5627.20 vs 5628.00)
2. Invoice Kundan (Vendor B): Discrepancy explanation (878.40 vs 1518.54)
"""

import pytest
from decimal import Decimal
from app.validators import InvoiceArithmeticValidator, ValidationStatus

def test_invoice_4531_fixed_numbers_reconciliation():
    # Fixed numbers from printed invoice WMP/B/04531
    # Line 1: 60 qty * 3700.00 = 222000.00, 2% discount = 4440.00 -> taxable = 217560.00, GST 5% = 10878.00
    # Line 2: 120 qty * 198.00 = 23760.00, 5% discount = 1188.00 -> taxable = 22572.00, GST 5% = 1128.60
    # Total Gross: 245760.00
    # Total Discount: 5628.00
    # Total Taxable: 240132.00
    # Total GST: 12006.60 (CGST 6003.30 + SGST 6003.30)
    # Round-off: +0.40
    # Invoice Amount: 252139.00

    # Old buggy calculation: Gross + GST - Roundoff - Invoice Amount
    buggy_discrepancy = (Decimal("245760.00") + Decimal("12006.60") - Decimal("0.40")) - Decimal("252139.00")
    assert buggy_discrepancy == Decimal("5627.20"), "Reproduced the 5627.20 bug!"

    # Correct discount value:
    total_discount = Decimal("4440.00") + Decimal("1188.00")
    assert total_discount == Decimal("5628.00")

    # Correct validator check:
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    doc_4531 = {
        "invoice_amount": "252139.00",
        "total_gst_amount": "12006.60",
        "round_off": "0.40",
        "items": [
            {"total_price": "222000.00", "Discount": "2.00"},
            {"total_price": "23760.00", "Discount": "5.00"}
        ]
    }
    res = validator.validate(doc_4531)
    assert res.status == ValidationStatus.PASS
    assert res.is_valid is True
    assert res.discrepancy == Decimal("0.00")
    assert res.reconciled_formula == "Taxable + GST + Freight + RoundOff"

def test_invoice_kundan_fixed_numbers_reconciliation():
    # Fixed numbers from printed Kundan invoice:
    # Gross items sum: 53542.60
    # Total Discount: 2141.704 (taxable: 51400.896)
    # Correct GST: 3020.08
    # Extracted GST: 2396.94 (short by 623.14)
    # Round-off: 0.01
    # Invoice Amount: 54421.00

    # Pipeline extraction error: Gross items + Extracted GST - Invoice Amount
    pipeline_extracted_sum = Decimal("53542.60") + Decimal("2396.94")
    raw_discrepancy = pipeline_extracted_sum - Decimal("54421.00")
    assert raw_discrepancy == Decimal("1518.54"), "Reproduced the 1518.54 discrepancy!"

    # Buggy partial offset (1518.54 - 640.14 = 878.40)
    assert raw_discrepancy - Decimal("640.14") == Decimal("878.40"), "Reproduced the 878.40 bug!"

    # On ground truth with correct GST (3020.08), the document reconciles:
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    doc_kundan_gt = {
        "invoice_amount": "54421.00",
        "total_gst_amount": "3020.08",
        "round_off": "0.01",
        "items": [
            {"total_price": "53542.60", "Discount": "4.00"}
        ]
    }
    res_gt = validator.validate(doc_kundan_gt)
    assert res_gt.status == ValidationStatus.PASS
    assert res_gt.is_valid is True
