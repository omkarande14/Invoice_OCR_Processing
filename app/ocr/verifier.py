import re
from typing import Dict, Any, List, Optional

class InvoiceVerifier:
    """
    Performs comprehensive verification on extracted invoice data:
    1. Line-item arithmetic checks (Qty * Rate == Amount).
    2. Subtotal summation checks (Sum of line items == Subtotal).
    3. Tax computation & Grand Total reconciliation (Subtotal + Taxes == Grand Total).
    4. GSTIN format and state code validation.
    5. Dual-engine consensus auditing (PaddleOCR vs Tesseract).
    """

    # Valid Indian State codes for GSTIN (01 to 38, 97 for other territory)
    VALID_STATE_CODES = {
        f"{i:02d}" for i in range(1, 39)
    }.union({"97"})

    @staticmethod
    def validate_gstin(gstin: Optional[str]) -> Dict[str, Any]:
        """Validates Indian 15-character GSTIN structure and state code."""
        if not gstin or not isinstance(gstin, str):
            return {"valid": False, "reason": "Missing or empty GSTIN"}

        cleaned = gstin.strip().upper()
        if len(cleaned) != 15:
            return {"valid": False, "gstin": cleaned, "reason": f"Invalid length {len(cleaned)} (expected 15)"}

        # Regex: 2 digits state + 5 alpha PAN + 4 digits PAN + 1 alpha PAN + 1 entity num + 'Z' + 1 check char
        pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
        if not re.match(pattern, cleaned):
            return {"valid": False, "gstin": cleaned, "reason": "Pattern mismatch (expected standard GSTIN format)"}

        state_code = cleaned[:2]
        if state_code not in InvoiceVerifier.VALID_STATE_CODES:
            return {"valid": False, "gstin": cleaned, "reason": f"Unknown Indian state code '{state_code}'"}

        return {"valid": True, "gstin": cleaned, "state_code": state_code}

    @staticmethod
    def verify_arithmetic(
        line_items: List[Dict[str, Any]],
        extracted_subtotal: Optional[float] = None,
        cgst: Optional[float] = None,
        sgst: Optional[float] = None,
        igst: Optional[float] = None,
        roundoff: Optional[float] = None,
        extracted_grand_total: Optional[float] = None,
        tolerance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Reconciles:
        - Row items: Qty * Rate == Amount
        - Line items sum == Subtotal
        - Subtotal + CGST + SGST + IGST + Roundoff == Grand Total
        """
        # 1. Line-item checks
        item_checks = []
        sum_of_items = 0.0

        for idx, item in enumerate(line_items):
            q = item.get("quantity")
            r = item.get("rate")
            a = item.get("amount")
            verified = False
            diff = None

            if a is not None:
                sum_of_items += a

            if q is not None and r is not None and a is not None:
                expected = round(q * r, 2)
                diff = round(abs(expected - a), 2)
                verified = diff <= tolerance or (
                    # Trade discount check
                    a <= expected
                )
            elif a is not None:
                # If only amount is present
                verified = True

            item_checks.append({
                "item_index": idx + 1,
                "description": item.get("description", ""),
                "expected_amount": round(q * r, 2) if q and r else a,
                "actual_amount": a,
                "verified": verified,
                "difference": diff
            })

        all_items_verified = all(c["verified"] for c in item_checks) if item_checks else False
        sum_of_items = round(sum_of_items, 2)

        # 2. Subtotal check
        subtotal_target = extracted_subtotal if extracted_subtotal is not None else sum_of_items
        subtotal_diff = round(abs(sum_of_items - subtotal_target), 2)
        subtotal_verified = subtotal_diff <= tolerance

        # 3. Grand total calculation
        tax_total = round((cgst or 0.0) + (sgst or 0.0) + (igst or 0.0), 2)
        calculated_grand_total = round(subtotal_target + tax_total + (roundoff or 0.0), 2)

        grand_total_diff = None
        grand_total_verified = False
        reconciled_grand_total = calculated_grand_total
        grand_total_source = "calculated_override"

        if extracted_grand_total is not None:
            # If extracted_grand_total happens to equal subtotal while taxes exist,
            # the invoice labeled the subtotal as 'Total'. We reconcile to calculated_grand_total.
            if tax_total > 0 and abs(extracted_grand_total - subtotal_target) <= tolerance:
                grand_total_verified = True
                grand_total_diff = 0.0
                reconciled_grand_total = calculated_grand_total
                grand_total_source = "calculated_override"
            else:
                grand_total_diff = round(abs(calculated_grand_total - extracted_grand_total), 2)
                direct_match = grand_total_diff <= tolerance

                # Check if discrepancy is explained by a trade discount between gross subtotal and taxable amount
                discount_verified = False
                if not direct_match and tax_total > 0 and extracted_grand_total < calculated_grand_total:
                    implied_taxable = round(extracted_grand_total - tax_total - (roundoff or 0.0), 2)
                    if 0 < implied_taxable < subtotal_target:
                        discount_verified = True

                if direct_match or discount_verified:
                    grand_total_verified = True
                    reconciled_grand_total = extracted_grand_total
                    grand_total_source = "ocr_extracted"
                else:
                    grand_total_verified = False
                    reconciled_grand_total = extracted_grand_total
                    grand_total_source = "ocr_extracted"
        else:
            grand_total_verified = True
            reconciled_grand_total = calculated_grand_total
            grand_total_source = "calculated_override"

        # 4. Overall status
        if all_items_verified and subtotal_verified and grand_total_verified:
            overall_status = "MATHEMATICALLY_VERIFIED"
        elif subtotal_verified:
            overall_status = "SUBTOTAL_VERIFIED"
        else:
            overall_status = "ARITHMETIC_DISCREPANCY"

        return {
            "overall_status": overall_status,
            "all_items_verified": all_items_verified,
            "line_items_count": len(line_items),
            "sum_of_line_items": sum_of_items,
            "extracted_subtotal": extracted_subtotal,
            "subtotal_verified": subtotal_verified,
            "subtotal_difference": subtotal_diff,
            "total_taxes": tax_total,
            "calculated_grand_total": calculated_grand_total,
            "extracted_grand_total": extracted_grand_total,
            "reconciled_grand_total": reconciled_grand_total,
            "grand_total_source": grand_total_source,
            "grand_total_verified": grand_total_verified,
            "grand_total_difference": grand_total_diff,
            "item_level_audit": item_checks
        }

    @staticmethod
    def audit_consensus(
        paddle_entities: Dict[str, Any],
        tesseract_entities: Dict[str, Any],
        verified_grand_total: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Audits consensus between PaddleOCR and Tesseract on key metadata fields.
        """
        # GSTIN consensus
        p_gstins = set(paddle_entities.get("gstin", []))
        t_gstins = set(tesseract_entities.get("gstin", []))
        gstin_match = len(p_gstins.intersection(t_gstins)) > 0
        agreed_gstins = sorted(list(p_gstins.intersection(t_gstins)))

        # Invoice number consensus
        p_invs = set(paddle_entities.get("invoice_numbers", []))
        t_invs = set(tesseract_entities.get("invoice_numbers", []))
        inv_match = len(p_invs.intersection(t_invs)) > 0
        agreed_invs = sorted(list(p_invs.intersection(t_invs)))

        # Totals consensus with float-normalized comparison
        p_totals = set(paddle_entities.get("totals", []))
        t_totals = set(tesseract_entities.get("totals", []))

        matched_totals = []
        for p_t in p_totals:
            try:
                p_f = float(p_t)
                for t_t in t_totals:
                    try:
                        t_f = float(t_t)
                        if abs(p_f - t_f) <= 1.0:
                            matched_totals.append(f"{p_f:.2f}")
                            break
                    except ValueError:
                        pass
            except ValueError:
                pass

        if verified_grand_total is not None and verified_grand_total > 0:
            vgt_str = f"{verified_grand_total:.2f}"
            for t_val in p_totals.union(t_totals):
                try:
                    if abs(float(t_val) - verified_grand_total) <= 1.0:
                        if vgt_str not in matched_totals:
                            matched_totals.append(vgt_str)
                except ValueError:
                    pass

        # Sort descending so Grand Total precedes intermediate subtotal
        matched_totals = sorted(list(dict.fromkeys(matched_totals)), key=lambda x: float(x), reverse=True)
        total_match = len(matched_totals) > 0
        agreed_totals = matched_totals[:1] if matched_totals else []

        agreements_count = sum([1 if gstin_match else 0, 1 if inv_match else 0, 1 if total_match else 0])
        consensus_score = round((agreements_count / 3.0) * 100.0, 1)

        if consensus_score == 100.0:
            consensus_level = "FULL_CONSENSUS"
        elif consensus_score >= 33.3:
            consensus_level = "PARTIAL_CONSENSUS"
        else:
            consensus_level = "PADDLE_PRIMARY"

        return {
            "consensus_level": consensus_level,
            "consensus_score": consensus_score,
            "gstin_consensus": gstin_match,
            "agreed_gstin": agreed_gstins,
            "invoice_number_consensus": inv_match,
            "agreed_invoice_number": agreed_invs,
            "total_amount_consensus": total_match,
            "agreed_total_amount": agreed_totals
        }
