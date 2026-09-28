# tests/test_validators.py
import pytest
import json
from pathlib import Path
from decimal import Decimal
from app.validators import (
    ValidationStatus,
    ValidationResult,
    InvoiceArithmeticValidator,
    classify_reconciliation
)

def test_validator_tristate_pass():
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    doc = {
        "invoice_amount": "118.00",
        "total_gst_amount": "18.00",
        "round_off": "0.00",
        "items": [
            {"total_price": "100.00", "Discount": "0.00"}
        ]
    }
    res = validator.validate(doc)
    assert res.status == ValidationStatus.PASS
    assert res.is_valid is True

def test_validator_tristate_fail():
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    doc = {
        "invoice_amount": "200.00", # Wrong total
        "total_gst_amount": "18.00",
        "round_off": "0.00",
        "items": [
            {"total_price": "100.00", "Discount": "0.00"}
        ]
    }
    res = validator.validate(doc)
    assert res.status == ValidationStatus.FAIL
    assert res.is_valid is False
    assert res.discrepancy > Decimal("50.0")

def test_validator_tristate_insufficient_data():
    validator = InvoiceArithmeticValidator()
    # Missing items and zero total
    res = validator.validate({"invoice_amount": None, "items": []})
    assert res.status == ValidationStatus.INSUFFICIENT_DATA
    assert res.is_valid is False

def test_validator_never_returns_true_on_exception():
    validator = InvoiceArithmeticValidator()
    # Malformed data
    res = validator.validate({"invoice_amount": "invalid_number", "items": "not_a_list"})
    assert res.status == ValidationStatus.INSUFFICIENT_DATA
    assert res.is_valid is False

def test_failing_invoices_reconciliation_diagnosis():
    # Analyze the 2 failing invoices: WMP/B/04531 and KP26264882
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    gt_dir = Path("data_private/ground_truth")
    if not gt_dir.exists():
        gt_dir = Path("data/ground_truth")

    p_4531 = gt_dir / "invoice_4531_expected.json"
    p_kundan = gt_dir / "kundan_expected.json"
    if not p_4531.exists() or not p_kundan.exists():
        pytest.skip("Private ground truth files not present on fresh clone.")

    with open(p_4531, "r", encoding="utf-8") as f:
        gt_4531 = json.load(f)

    res_4531 = validator.validate(gt_4531)
    assert res_4531.status == ValidationStatus.PASS
    assert res_4531.is_valid is True

    with open(p_kundan, "r", encoding="utf-8") as f:
        gt_kundan = json.load(f)

    res_kundan = validator.validate(gt_kundan)
    assert res_kundan.status == ValidationStatus.PASS
    assert res_kundan.is_valid is True
