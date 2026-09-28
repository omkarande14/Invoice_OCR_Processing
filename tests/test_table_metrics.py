# tests/test_table_metrics.py
import pytest
from app.rrsi.table_metrics import (
    evaluate_table,
    TableEvaluationResult,
    TableStructureError
)

def test_table_metrics_perfect_match():
    exp = [
        {"description": "ALBUCEL 20%", "hsn_sac": "30021290", "quantity": 60, "unit_price": "3700.00", "total_price": "222000.00"},
        {"description": "TELIPSA INJ", "hsn_sac": "30049099", "quantity": 120, "unit_price": "198.00", "total_price": "23760.00"}
    ]
    pred = [
        {"description": "ALBUCEL 20%", "hsn_sac": "30021290", "quantity": 60, "unit_price": "3700.00", "total_price": "222000.00"},
        {"description": "TELIPSA INJ", "hsn_sac": "30049099", "quantity": 120, "unit_price": "198.00", "total_price": "23760.00"}
    ]

    res = evaluate_table(exp, pred)
    assert res.row_precision == 1.0
    assert res.row_recall == 1.0
    assert res.row_f1 == 1.0
    assert res.cell_f1 == 1.0
    assert len(res.structure_errors) == 0

def test_table_metrics_guard_none_none_matches():
    # If both expected and predicted have None/empty cells, it must NOT count as a positive cell match
    exp = [{"description": "Item 1", "hsn_sac": None, "unit": None, "quantity": 10, "unit_price": "50.00", "total_price": "500.00"}]
    pred = [{"description": "Item 1", "hsn_sac": None, "unit": None, "quantity": 10, "unit_price": "50.00", "total_price": "500.00"}]

    res = evaluate_table(exp, pred)
    # Expected cells with values: description, quantity, unit_price, total_price (4 cells)
    # None cells should not artificially inflate cell matches
    assert res.total_expected_cells == 4
    assert res.correct_cells == 4
    assert res.cell_recall == 1.0

def test_table_metrics_row_split_and_merge():
    # Ground truth: 1 row with qty 10, total 1000
    # Prediction: 2 rows with qty 5, total 500 each (row split)
    exp = [
        {"description": "Item Alpha", "hsn_sac": "3004", "quantity": 10, "unit_price": "100.00", "total_price": "1000.00"}
    ]
    pred = [
        {"description": "Item Alpha", "hsn_sac": "3004", "quantity": 5, "unit_price": "100.00", "total_price": "500.00"},
        {"description": "Item Alpha", "hsn_sac": "3004", "quantity": 5, "unit_price": "100.00", "total_price": "500.00"}
    ]

    res = evaluate_table(exp, pred)
    assert res.row_recall == 1.0  # One matched
    assert res.row_precision == 0.5 # 1 out of 2
    assert any(err.error_type == TableStructureError.ROW_SPLIT for err in res.structure_errors)

def test_table_metrics_column_shift():
    # Rate shifted into HSN column
    exp = [
        {"description": "Item Beta", "hsn_sac": "30049099", "quantity": 10, "unit_price": "150.00", "total_price": "1500.00"}
    ]
    pred = [
        {"description": "Item Beta", "hsn_sac": "150.00", "quantity": 10, "unit_price": None, "total_price": "1500.00"}
    ]

    res = evaluate_table(exp, pred)
    assert any(err.error_type == TableStructureError.COLUMN_SHIFT for err in res.structure_errors)
