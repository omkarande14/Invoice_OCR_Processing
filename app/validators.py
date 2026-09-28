# app/validators.py
"""
Cross-Field Arithmetic and Accounting Consistency Validators for Invoice OCR.
Implements tri-state validation: PASS, FAIL, INSUFFICIENT_DATA.
Never returns True on exception or missing data.
Handles discounts, freight, cess, round-off, and GST tax structures.
Separates document internal inconsistency from pipeline extraction errors.
"""

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.rrsi.normalize import normalize_amount


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ValidationResult:
    status: ValidationStatus
    is_valid: bool
    items_sum: Decimal
    taxable_sum: Decimal
    discount_total: Decimal
    gst_amount: Decimal
    freight_amount: Decimal
    round_off: Decimal
    invoice_amount: Decimal
    discrepancy: Decimal
    reconciled_formula: Optional[str]
    message: str


def _to_decimal(val: Any, default: str = "0.00") -> Decimal:
    """Safely converts any value to Decimal."""
    if val is None:
        return Decimal(default)
    dec = normalize_amount(val)
    return dec if dec is not None else Decimal(default)


class InvoiceArithmeticValidator:
    """
    Validates cross-field arithmetic consistency across line-items, taxes, discounts,
    freight, round-off, and grand total.
    """

    def __init__(
        self,
        tolerance: Decimal = Decimal("1.00"),
        relative_tolerance: Decimal = Decimal("0.001")
    ):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance

    def validate(self, doc: Dict[str, Any]) -> ValidationResult:
        """
        Validates the arithmetic integrity of an invoice dictionary.
        Returns tri-state ValidationResult (PASS, FAIL, or INSUFFICIENT_DATA).
        """
        if not isinstance(doc, dict):
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_DATA,
                is_valid=False,
                items_sum=Decimal("0.00"),
                taxable_sum=Decimal("0.00"),
                discount_total=Decimal("0.00"),
                gst_amount=Decimal("0.00"),
                freight_amount=Decimal("0.00"),
                round_off=Decimal("0.00"),
                invoice_amount=Decimal("0.00"),
                discrepancy=Decimal("0.00"),
                reconciled_formula=None,
                message="Document is not a dictionary"
            )

        try:
            raw_inv_amt = doc.get("invoice_amount")
            items = doc.get("items")

            # Check for insufficient data
            if raw_inv_amt is None and (not items or not isinstance(items, list)):
                return ValidationResult(
                    status=ValidationStatus.INSUFFICIENT_DATA,
                    is_valid=False,
                    items_sum=Decimal("0.00"),
                    taxable_sum=Decimal("0.00"),
                    discount_total=Decimal("0.00"),
                    gst_amount=Decimal("0.00"),
                    freight_amount=Decimal("0.00"),
                    round_off=Decimal("0.00"),
                    invoice_amount=Decimal("0.00"),
                    discrepancy=Decimal("0.00"),
                    reconciled_formula=None,
                    message="Missing invoice_amount and line items"
                )

            inv_amt = _to_decimal(raw_inv_amt)
            if inv_amt <= Decimal("0.00"):
                return ValidationResult(
                    status=ValidationStatus.INSUFFICIENT_DATA,
                    is_valid=False,
                    items_sum=Decimal("0.00"),
                    taxable_sum=Decimal("0.00"),
                    discount_total=Decimal("0.00"),
                    gst_amount=Decimal("0.00"),
                    freight_amount=Decimal("0.00"),
                    round_off=Decimal("0.00"),
                    invoice_amount=inv_amt,
                    discrepancy=Decimal("0.00"),
                    reconciled_formula=None,
                    message="Invoice amount is zero or unparseable"
                )

            # GST amounts
            raw_gst = doc.get("total_gst_amount")
            if raw_gst is not None:
                gst_amt = _to_decimal(raw_gst)
            else:
                cgst = _to_decimal(doc.get("total_cgst_amount"))
                sgst = _to_decimal(doc.get("total_sgst_amount"))
                igst = _to_decimal(doc.get("total_igst_amount"))
                gst_amt = cgst + sgst + igst

            # Round off & Freight
            round_off = _to_decimal(doc.get("round_off"))
            freight = _to_decimal(doc.get("freight_amount") or doc.get("freight") or doc.get("shipping"))

            # Calculate item sums and discounts
            items_sum = Decimal("0.00")
            taxable_sum = Decimal("0.00")
            discount_total = Decimal("0.00")

            if isinstance(items, list) and len(items) > 0:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    p = _to_decimal(item.get("total_price") or item.get("amount") or item.get("price"))
                    items_sum += p

                    # Item level discount
                    disc_pct = _to_decimal(item.get("Discount") or item.get("discount_rate"))
                    disc_amt = _to_decimal(item.get("discount_amount") or item.get("discount"))
                    if disc_amt > Decimal("0.00"):
                        discount_total += disc_amt
                        item_taxable = max(Decimal("0.00"), p - disc_amt)
                    elif disc_pct > Decimal("0.00"):
                        computed_disc = p * (disc_pct / Decimal("100.00"))
                        discount_total += computed_disc
                        item_taxable = p - computed_disc
                    else:
                        item_taxable = p

                    t_val = item.get("taxable_value")
                    if t_val is not None:
                        taxable_sum += _to_decimal(t_val)
                    else:
                        taxable_sum += item_taxable
            else:
                # If no line items, taxable sum defaults to items_sum
                taxable_sum = items_sum

            # Allowed tolerance check
            allowed_tol = max(self.tolerance, self.relative_tolerance * inv_amt)

            # Candidate accounting formulas in Indian GST:
            formulas = [
                # 1. Taxable + GST + Freight + Roundoff
                ("Taxable + GST + Freight + RoundOff", abs((taxable_sum + gst_amt + freight + round_off) - inv_amt)),
                # 2. Taxable + GST + Freight - Roundoff
                ("Taxable + GST + Freight - RoundOff", abs((taxable_sum + gst_amt + freight - round_off) - inv_amt)),
                # 3. Gross Items + GST + Freight + Roundoff
                ("Gross Items + GST + Freight + RoundOff", abs((items_sum + gst_amt + freight + round_off) - inv_amt)),
                # 4. Gross Items + GST + Freight - Roundoff
                ("Gross Items + GST + Freight - RoundOff", abs((items_sum + gst_amt + freight - round_off) - inv_amt)),
                # 5. (Gross Items - Discount) + GST + Roundoff
                ("(Gross Items - Disc) + GST + RoundOff", abs(((items_sum - discount_total) + gst_amt + freight + round_off) - inv_amt)),
                # 6. (Gross Items - Discount) + GST - Roundoff
                ("(Gross Items - Disc) + GST - RoundOff", abs(((items_sum - discount_total) + gst_amt + freight - round_off) - inv_amt)),
                # 7. Tax inclusive / net items sum
                ("Tax-Inclusive Items Sum", abs(items_sum - inv_amt)),
                # 8. Taxable sum equals grand total (zero GST)
                ("Zero GST Taxable Sum", abs(taxable_sum - inv_amt))
            ]

            # Find best match
            best_formula, min_diff = min(formulas, key=lambda x: x[1])

            if min_diff <= allowed_tol:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    is_valid=True,
                    items_sum=items_sum,
                    taxable_sum=taxable_sum,
                    discount_total=discount_total,
                    gst_amount=gst_amt,
                    freight_amount=freight,
                    round_off=round_off,
                    invoice_amount=inv_amt,
                    discrepancy=min_diff,
                    reconciled_formula=best_formula,
                    message=f"Reconciled via '{best_formula}' (diff {min_diff:.2f} <= tol {allowed_tol:.2f})"
                )

            return ValidationResult(
                status=ValidationStatus.FAIL,
                is_valid=False,
                items_sum=items_sum,
                taxable_sum=taxable_sum,
                discount_total=discount_total,
                gst_amount=gst_amt,
                freight_amount=freight,
                round_off=round_off,
                invoice_amount=inv_amt,
                discrepancy=min_diff,
                reconciled_formula=None,
                message=f"Arithmetic discrepancy {min_diff:.2f} exceeds tolerance {allowed_tol:.2f} "
                        f"(Items: {items_sum}, Taxable: {taxable_sum}, GST: {gst_amt}, Total: {inv_amt})"
            )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_DATA,
                is_valid=False,
                items_sum=Decimal("0.00"),
                taxable_sum=Decimal("0.00"),
                discount_total=Decimal("0.00"),
                gst_amount=Decimal("0.00"),
                freight_amount=Decimal("0.00"),
                round_off=Decimal("0.00"),
                invoice_amount=Decimal("0.00"),
                discrepancy=Decimal("0.00"),
                reconciled_formula=None,
                message=f"Validation failed with error: {e}"
            )


def classify_reconciliation(
    prediction: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    tolerance: float = 1.0
) -> Tuple[str, ValidationResult, Optional[ValidationResult]]:
    """
    Classifies arithmetic reconciliation into:
    - 'consistent': Prediction reconciles mathematically.
    - 'document_inconsistent': Prediction fails, but Ground Truth is also inconsistent on paper.
    - 'pipeline_wrong': Ground Truth reconciles on paper, but Pipeline extraction failed.

    Returns:
        (classification_code, pred_result, gt_result)
    """
    validator = InvoiceArithmeticValidator(tolerance=Decimal(str(tolerance)))
    pred_res = validator.validate(prediction)

    if pred_res.status == ValidationStatus.PASS:
        return "consistent", pred_res, None

    if ground_truth:
        gt_res = validator.validate(ground_truth)
        if gt_res.status != ValidationStatus.PASS:
            return "document_inconsistent", pred_res, gt_res
        return "pipeline_wrong", pred_res, gt_res

    return "pipeline_wrong", pred_res, None
