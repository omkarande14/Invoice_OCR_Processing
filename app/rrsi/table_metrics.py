# app/rrsi/table_metrics.py
"""
Table & Line-Item Extraction Evaluation Engine.
1. Optimal 1-to-1 bipartite matching using scipy.optimize.linear_sum_assignment.
2. Row-level Precision, Recall, and F1.
3. Cell-level Precision, Recall, and F1 across (description, hsn, qty, unit, rate, amount, tax).
4. Structural error classification: row_merge, row_split, column_shift, missing_row, extra_row.
5. Strict guard preventing None == None from being treated as true matches.
"""

from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.rrsi.normalize import (
    normalize_vendor_name,
    match_amount,
    normalize_amount
)


class TableStructureError(str, Enum):
    ROW_MERGE = "table_row_merge"
    ROW_SPLIT = "table_row_split"
    COLUMN_SHIFT = "table_column_shift"
    MISSING_ROW = "table_missing_row"
    EXTRA_ROW = "table_extra_row"


@dataclass
class TableStructureAnomaly:
    error_type: TableStructureError
    message: str
    exp_idx: Optional[int] = None
    pred_idx: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableEvaluationResult:
    row_precision: float
    row_recall: float
    row_f1: float
    cell_precision: float
    cell_recall: float
    cell_f1: float
    matched_row_count: int
    total_expected_rows: int
    total_predicted_rows: int
    total_expected_cells: int
    correct_cells: int
    total_predicted_cells: int
    structure_errors: List[TableStructureAnomaly] = field(default_factory=list)


CELL_FIELDS = [
    ("description", ["description", "item_name", "item_description", "name"]),
    ("hsn", ["hsn_sac", "hsn", "sac"]),
    ("qty", ["quantity", "qty"]),
    ("unit", ["unit", "Pack", "pack", "uom"]),
    ("rate", ["unit_price", "rate", "price"]),
    ("amount", ["total_price", "amount", "item_amount"]),
    ("tax", ["Gst%", "gst_rate", "tax_rate", "GST_AMT", "cgst_amount", "sgst_amount"])
]


def _extract_cell_value(row: Dict[str, Any], canonical_key: str) -> Optional[Any]:
    """Retrieves cell value using canonical aliases."""
    for key, aliases in CELL_FIELDS:
        if key == canonical_key:
            for alias in aliases:
                val = row.get(alias)
                if val is not None and str(val).strip() != "":
                    return val
            return None
    return None


def _compute_row_similarity(exp: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """Computes a normalized similarity [0.0, 1.0] between an expected and predicted row."""
    score = 0.0
    weights_total = 0.0

    # 1. Description (weight: 0.35)
    exp_desc = _extract_cell_value(exp, "description")
    pred_desc = _extract_cell_value(pred, "description")
    if exp_desc and pred_desc:
        weights_total += 0.35
        n1 = normalize_vendor_name(exp_desc)
        n2 = normalize_vendor_name(pred_desc)
        t1, t2 = set(n1.split()), set(n2.split())
        if t1 and t2:
            sim = 2.0 * len(t1 & t2) / (len(t1) + len(t2))
            score += 0.35 * sim

    # 2. HSN (weight: 0.15)
    exp_hsn = _extract_cell_value(exp, "hsn")
    pred_hsn = _extract_cell_value(pred, "hsn")
    if exp_hsn and pred_hsn:
        weights_total += 0.15
        if str(exp_hsn).strip().replace(" ", "") == str(pred_hsn).strip().replace(" ", ""):
            score += 0.15

    # 3. Quantity (weight: 0.15)
    exp_qty = _extract_cell_value(exp, "qty")
    pred_qty = _extract_cell_value(pred, "qty")
    if exp_qty is not None and pred_qty is not None:
        weights_total += 0.15
        q1, q2 = normalize_amount(exp_qty), normalize_amount(pred_qty)
        if q1 is not None and q2 is not None and abs(q1 - q2) <= Decimal("0.05"):
            score += 0.15

    # 4. Rate (weight: 0.15)
    exp_rate = _extract_cell_value(exp, "rate")
    pred_rate = _extract_cell_value(pred, "rate")
    if exp_rate is not None and pred_rate is not None:
        weights_total += 0.15
        matched, _ = match_amount(exp_rate, pred_rate, rel_tol=Decimal("0.01"), abs_tol=Decimal("1.00"))
        if matched:
            score += 0.15

    # 5. Amount (weight: 0.20)
    exp_amt = _extract_cell_value(exp, "amount")
    pred_amt = _extract_cell_value(pred, "amount")
    if exp_amt is not None and pred_amt is not None:
        weights_total += 0.20
        matched, _ = match_amount(exp_amt, pred_amt, rel_tol=Decimal("0.01"), abs_tol=Decimal("1.00"))
        if matched:
            score += 0.20

    if weights_total == 0.0:
        return 0.0
    # Normalize score by present attributes
    return float(score / weights_total)


def _check_cell_match(canonical_key: str, exp_val: Any, pred_val: Any) -> bool:
    """Checks whether two cells match, strictly guarding None == None."""
    if exp_val is None or pred_val is None:
        return False
    s_exp = str(exp_val).strip()
    s_pred = str(pred_val).strip()
    if not s_exp or not s_pred:
        return False

    if canonical_key in ("qty", "rate", "amount", "tax"):
        matched, _ = match_amount(s_exp, s_pred, rel_tol=Decimal("0.01"), abs_tol=Decimal("1.00"))
        return matched

    if canonical_key == "description":
        n1 = normalize_vendor_name(s_exp)
        n2 = normalize_vendor_name(s_pred)
        t1, t2 = set(n1.split()), set(n2.split())
        return (2.0 * len(t1 & t2) / (len(t1) + len(t2))) >= 0.70 if (t1 and t2) else False

    # HSN / Unit
    return s_exp.lower().replace(" ", "") == s_pred.lower().replace(" ", "")


def evaluate_table(
    expected_rows: List[Dict[str, Any]],
    predicted_rows: List[Dict[str, Any]],
    similarity_threshold: float = 0.40
) -> TableEvaluationResult:
    """
    Evaluates extracted line-items against ground-truth items using optimal 1-to-1 matching.
    """
    m = len(expected_rows)
    n = len(predicted_rows)
    structure_errors: List[TableStructureAnomaly] = []

    if m == 0 and n == 0:
        return TableEvaluationResult(
            row_precision=1.0, row_recall=1.0, row_f1=1.0,
            cell_precision=1.0, cell_recall=1.0, cell_f1=1.0,
            matched_row_count=0, total_expected_rows=0, total_predicted_rows=0,
            total_expected_cells=0, correct_cells=0, total_predicted_cells=0
        )

    if m == 0 and n > 0:
        for j in range(n):
            structure_errors.append(TableStructureAnomaly(
                error_type=TableStructureError.EXTRA_ROW,
                message=f"Extra predicted row {j} when ground truth is empty",
                pred_idx=j
            ))
        return TableEvaluationResult(
            row_precision=0.0, row_recall=1.0, row_f1=0.0,
            cell_precision=0.0, cell_recall=1.0, cell_f1=0.0,
            matched_row_count=0, total_expected_rows=0, total_predicted_rows=n,
            total_expected_cells=0, correct_cells=0, total_predicted_cells=n * len(CELL_FIELDS),
            structure_errors=structure_errors
        )

    if m > 0 and n == 0:
        for i in range(m):
            structure_errors.append(TableStructureAnomaly(
                error_type=TableStructureError.MISSING_ROW,
                message=f"Missing expected row {i}",
                exp_idx=i
            ))
        return TableEvaluationResult(
            row_precision=1.0, row_recall=0.0, row_f1=0.0,
            cell_precision=1.0, cell_recall=0.0, cell_f1=0.0,
            matched_row_count=0, total_expected_rows=m, total_predicted_rows=0,
            total_expected_cells=m * len(CELL_FIELDS), correct_cells=0, total_predicted_cells=0,
            structure_errors=structure_errors
        )

    # Cost matrix for bipartite matching: cost = 1.0 - similarity
    cost_matrix = np.zeros((m, n), dtype=float)
    sim_matrix = np.zeros((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            sim = _compute_row_similarity(expected_rows[i], predicted_rows[j])
            sim_matrix[i, j] = sim
            cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: List[Tuple[int, int]] = []
    matched_exp_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= similarity_threshold:
            matched_pairs.append((r, c))
            matched_exp_indices.add(r)
            matched_pred_indices.add(c)

    # Detect unassigned rows
    unassigned_exp = set(range(m)) - matched_exp_indices
    unassigned_pred = set(range(n)) - matched_pred_indices

    # Check for row merge: multiple expected items combined into one predicted item
    # e.g. sum of quantities or amounts of two unassigned expected rows matches a predicted row
    for p_idx in range(n):
        pred_qty = normalize_amount(_extract_cell_value(predicted_rows[p_idx], "qty"))
        pred_amt = normalize_amount(_extract_cell_value(predicted_rows[p_idx], "amount"))
        if pred_qty and len(unassigned_exp) >= 2:
            # Check combinations of two expected rows
            exp_list = list(unassigned_exp)
            for i1 in range(len(exp_list)):
                for i2 in range(i1 + 1, len(exp_list)):
                    e1_qty = normalize_amount(_extract_cell_value(expected_rows[exp_list[i1]], "qty")) or Decimal("0")
                    e2_qty = normalize_amount(_extract_cell_value(expected_rows[exp_list[i2]], "qty")) or Decimal("0")
                    if abs((e1_qty + e2_qty) - pred_qty) <= Decimal("0.05") and (e1_qty > 0 and e2_qty > 0):
                        structure_errors.append(TableStructureAnomaly(
                            error_type=TableStructureError.ROW_MERGE,
                            message=f"Rows {exp_list[i1]} and {exp_list[i2]} merged into predicted row {p_idx}",
                            pred_idx=p_idx,
                            details={"merged_exp_indices": [exp_list[i1], exp_list[i2]]}
                        ))

    # Check for row split: one expected item split across multiple predicted rows
    for e_idx in range(m):
        exp_qty = normalize_amount(_extract_cell_value(expected_rows[e_idx], "qty"))
        exp_amt = normalize_amount(_extract_cell_value(expected_rows[e_idx], "amount"))
        exp_desc = str(_extract_cell_value(expected_rows[e_idx], "description") or "").lower()

        # Find predicted rows that share the same description
        matching_preds = [
            j for j in range(n)
            if exp_desc and exp_desc[:8] in str(_extract_cell_value(predicted_rows[j], "description") or "").lower()
        ]
        if len(matching_preds) >= 2:
            sum_pred_qty = sum(
                (normalize_amount(_extract_cell_value(predicted_rows[j], "qty")) or Decimal("0"))
                for j in matching_preds
            )
            if exp_qty and abs(sum_pred_qty - exp_qty) <= Decimal("0.05"):
                structure_errors.append(TableStructureAnomaly(
                    error_type=TableStructureError.ROW_SPLIT,
                    message=f"Expected row {e_idx} split into predicted rows {matching_preds}",
                    exp_idx=e_idx,
                    details={"split_pred_indices": matching_preds}
                ))

    # Check for column shift: e.g. rate value found in HSN or amount column
    for r, c in matched_pairs:
        exp_r = expected_rows[r]
        pred_c = predicted_rows[c]

        exp_rate = _extract_cell_value(exp_r, "rate")
        pred_hsn = _extract_cell_value(pred_c, "hsn")
        if exp_rate and pred_hsn:
            m_shift, _ = match_amount(exp_rate, pred_hsn, rel_tol=Decimal("0.01"), abs_tol=Decimal("0.05"))
            if m_shift:
                structure_errors.append(TableStructureAnomaly(
                    error_type=TableStructureError.COLUMN_SHIFT,
                    message=f"Column shift: Expected rate '{exp_rate}' found in HSN column in row {c}",
                    exp_idx=r,
                    pred_idx=c
                ))

    # Missing & extra rows
    for r in unassigned_exp:
        structure_errors.append(TableStructureAnomaly(
            error_type=TableStructureError.MISSING_ROW,
            message=f"Expected row {r} not extracted",
            exp_idx=r
        ))
    for c in unassigned_pred:
        structure_errors.append(TableStructureAnomaly(
            error_type=TableStructureError.EXTRA_ROW,
            message=f"Extra extracted row {c}",
            pred_idx=c
        ))

    # Row metrics
    matched_row_count = len(matched_pairs)
    row_p = float(matched_row_count / n) if n > 0 else 1.0
    row_r = float(matched_row_count / m) if m > 0 else 1.0
    row_f1 = float(2.0 * row_p * row_r / (row_p + row_r)) if (row_p + row_r) > 0 else 0.0

    # Cell-level evaluation: strictly evaluate non-null expected cells
    total_exp_cells = 0
    correct_cells = 0
    total_pred_cells = 0

    # Count predicted cells present
    for j in range(n):
        for key, _ in CELL_FIELDS:
            if _extract_cell_value(predicted_rows[j], key) is not None:
                total_pred_cells += 1

    # Count expected cells present
    for i in range(m):
        for key, _ in CELL_FIELDS:
            if _extract_cell_value(expected_rows[i], key) is not None:
                total_exp_cells += 1

    # Count matched cells in assigned pairs
    for r, c in matched_pairs:
        exp_r = expected_rows[r]
        pred_c = predicted_rows[c]
        for key, _ in CELL_FIELDS:
            e_val = _extract_cell_value(exp_r, key)
            p_val = _extract_cell_value(pred_c, key)
            if e_val is not None:
                if _check_cell_match(key, e_val, p_val):
                    correct_cells += 1

    cell_p = float(correct_cells / total_pred_cells) if total_pred_cells > 0 else (1.0 if total_exp_cells == 0 else 0.0)
    cell_r = float(correct_cells / total_exp_cells) if total_exp_cells > 0 else 1.0
    cell_f1 = float(2.0 * cell_p * cell_r / (cell_p + cell_r)) if (cell_p + cell_r) > 0 else 0.0

    return TableEvaluationResult(
        row_precision=round(row_p, 4),
        row_recall=round(row_r, 4),
        row_f1=round(row_f1, 4),
        cell_precision=round(cell_p, 4),
        cell_recall=round(cell_r, 4),
        cell_f1=round(cell_f1, 4),
        matched_row_count=matched_row_count,
        total_expected_rows=m,
        total_predicted_rows=n,
        total_expected_cells=total_exp_cells,
        correct_cells=correct_cells,
        total_predicted_cells=total_pred_cells,
        structure_errors=structure_errors
    )
