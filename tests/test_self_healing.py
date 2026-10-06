# tests/test_self_healing.py
"""
Unit tests for the Active Mathematical Self-Healing Engine (SelfHealingEngine)
and Semantic Synonym Ontology (SemanticOntology).
"""

import pytest
from app.ocr.self_healing import SelfHealingEngine
from app.ocr.semantic_ontology import (
    matches_any_alias,
    parse_credit_terms_days,
    PAYABLE_TOTAL_ALIASES,
    DUE_DATE_ALIASES,
    BUYER_ANCHORS
)

def test_semantic_ontology_matching():
    assert matches_any_alias("TO PAY: 34327.00", PAYABLE_TOTAL_ALIASES)
    assert matches_any_alias("Net Payable Amount", PAYABLE_TOTAL_ALIASES)
    assert matches_any_alias("Due Dt. 10/08/2026", DUE_DATE_ALIASES)
    assert matches_any_alias("Details of Receiver (Billed to)", BUYER_ANCHORS)
    assert matches_any_alias("Random Unrelated Text", PAYABLE_TOTAL_ALIASES) is False

def test_credit_terms_parsing():
    assert parse_credit_terms_days("Credit: 30 Days") == 30
    assert parse_credit_terms_days("Terms: 7 days net") == 7
    assert parse_credit_terms_days("Payment due within 15 days of invoice") == 15
    assert parse_credit_terms_days("Immediate Cash Payment") is None

def test_line_item_math_derivation():
    raw_items = [
        {
            "description": "Medicine A",
            "quantity": 10,
            "free_quantity": 2,
            "rate": 100.0,
            "amount": None,  # Missing
            "discount_pct": 10.0,
            "cgst_rate": 2.5,
            "sgst_rate": 2.5
        },
        {
            "description": "Medicine B",
            "quantity": 5,
            "free_quantity": 0,
            "rate": None,  # Missing
            "amount": 250.0,
            "discount_pct": 0.0,
            "cgst_rate": 2.5,
            "sgst_rate": 2.5
        }
    ]

    res = SelfHealingEngine.heal(
        meta={"invoice_date": "01/08/2026"},
        fin={},
        raw_items=raw_items
    )

    items = res["items"]
    # Item 1: 10 * 100 * (1 - 0.10) = 900.0
    assert items[0]["amount"] == 900.0
    assert items[0]["total_quantity"] == 12
    assert items[0]["free_item_yn"] == "1"
    assert items[0]["cgst_amount"] == 22.50
    assert items[0]["sgst_amount"] == 22.50

    # Item 2: 250 / 5 = 50.0
    assert items[1]["rate"] == 50.0
    assert items[1]["total_quantity"] == 5
    assert items[1]["free_item_yn"] == "0"
    assert items[1]["cgst_amount"] == 6.25
    assert items[1]["sgst_amount"] == 6.25

def test_grand_total_and_tax_auto_deduction():
    raw_items = [
        {
            "description": "Item 1",
            "quantity": 10,
            "rate": 100.0,
            "amount": 1000.0,
            "cgst_amount": 25.0,
            "sgst_amount": 25.0
        },
        {
            "description": "Item 2",
            "quantity": 5,
            "rate": 200.0,
            "amount": 1000.0,
            "cgst_amount": 25.0,
            "sgst_amount": 25.0
        }
    ]

    # Case A: Missing grand_total, cgst, sgst
    res = SelfHealingEngine.heal(
        meta={"invoice_date": "01/08/2026"},
        fin={"roundoff": 0.50},  # subtotal and grand_total missing
        raw_items=raw_items
    )

    fin = res["fin"]
    assert fin["cgst"] == 50.0
    assert fin["sgst"] == 50.0
    # subtotal = 2000.0, taxes = 100.0, roundoff = 0.50 -> 2100.50
    assert fin["grand_total"] == 2100.50

def test_mislabeled_subtotal_grand_total_reconciliation():
    raw_items = [
        {
            "description": "Item 1",
            "quantity": 10,
            "rate": 100.0,
            "amount": 1000.0,
            "cgst_amount": 25.0,
            "sgst_amount": 25.0
        }
    ]

    # Case B: extracted grand_total is 1000.0 (the taxable subtotal), but taxes are 50.0
    res = SelfHealingEngine.heal(
        meta={"invoice_date": "01/08/2026"},
        fin={
            "subtotal": 1000.0,
            "cgst": 25.0,
            "sgst": 25.0,
            "roundoff": 0.0,
            "grand_total": 1000.0  # Mislabeled
        },
        raw_items=raw_items
    )

    fin = res["fin"]
    # Must be corrected to 1050.0
    assert fin["grand_total"] == 1050.0

def test_due_date_not_synthesized_from_credit_terms():
    lines = [
        {"text": "Invoice Date: 10/08/2026"},
        {"text": "Payment Terms: Credit 20 Days"},
        {"text": "Thank you for your business"}
    ]

    res = SelfHealingEngine.heal(
        meta={"invoice_date": "10/08/2026", "due_date": None},
        fin={},
        raw_items=[],
        all_ocr_lines=lines
    )

    meta = res["meta"]
    # Due date must NOT be synthesized from credit terms; only reported when explicitly printed
    assert meta["due_date"] is None
