# app/ocr/self_healing.py
"""
Active Mathematical Self-Healing Engine.
Enforces double-entry accounting invariants, derives missing fields,
and auto-repairs discrepancies across invoice headers, taxes, and line items.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from app.ocr.semantic_ontology import parse_credit_terms_days

class SelfHealingEngine:
    """
    Active ledger constraint solver that repairs and derives invoice data:
    1. Line-item math completion (Qty * Rate * (1 - Disc) == Amount).
    2. Header total quantity and tax rollups.
    3. Grand total auto-deduction (Subtotal + Taxes + Roundoff == Grand Total).
    4. Due date inference from credit terms.
    5. GSTIN recovery from unassigned text tokens.
    """

    @classmethod
    def heal(
        cls,
        meta: Dict[str, Any],
        fin: Dict[str, Any],
        raw_items: List[Dict[str, Any]],
        all_ocr_lines: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes active self-healing across metadata, financial totals, and line items.
        Returns repaired (meta, fin, healed_items, healing_log).
        """
        healing_log: List[str] = []
        vendor = dict(meta.get("vendor", {}))
        buyer = dict(meta.get("buyer", {}))
        meta_repaired = dict(meta)
        fin_repaired = dict(fin)

        # ---------------------------------------------------------------------
        # 1. Line-Item Self-Healing
        # ---------------------------------------------------------------------
        healed_items = []
        for idx, item in enumerate(raw_items):
            h_item = dict(item)
            q = h_item.get("quantity")
            fq = h_item.get("free_quantity", 0) or 0
            r = h_item.get("rate")
            a = h_item.get("amount")
            disc = h_item.get("discount_pct") or 0.0

            # Derive free_item_yn
            if fq > 0:
                h_item["free_item_yn"] = "1"
            elif "free_item_yn" not in h_item or h_item["free_item_yn"] is None:
                h_item["free_item_yn"] = "0"

            # Derive total_quantity
            if q is not None:
                h_item["total_quantity"] = q + fq
            elif fq > 0:
                h_item["total_quantity"] = fq

            # Derive amount from qty and rate if missing
            if (a is None or float(a) == 0.0) and q is not None and r is not None:
                disc_factor = 1.0 - (float(disc) / 100.0) if disc else 1.0
                derived_amount = round(float(q) * float(r) * disc_factor, 2)
                h_item["amount"] = derived_amount
                healing_log.append(f"Item #{idx+1}: Derived missing amount {derived_amount} from qty {q} * rate {r}")

            # Derive rate from amount and qty if missing
            elif (r is None or float(r) == 0.0) and a is not None and q is not None and float(q) > 0:
                disc_factor = 1.0 - (float(disc) / 100.0) if disc else 1.0
                if disc_factor > 0:
                    derived_rate = round(float(a) / (float(q) * disc_factor), 2)
                    h_item["rate"] = derived_rate
                    healing_log.append(f"Item #{idx+1}: Derived missing rate {derived_rate} from amount {a} / qty {q}")

            # Reconcile misread quantity if rate and amount are present and qty is inconsistent
            elif q is not None and r is not None and a is not None and float(r) > 0 and float(a) > 0:
                disc_factor = 1.0 - (float(disc) / 100.0) if disc else 1.0
                if disc_factor > 0:
                    expected_a = round(float(q) * float(r) * disc_factor, 2)
                    if abs(expected_a - float(a)) > 1.0:
                        calc_q = round(float(a) / (float(r) * disc_factor))
                        if calc_q > 0 and abs(calc_q * float(r) * disc_factor - float(a)) < 0.05:
                            if calc_q != q and (str(int(calc_q)) in str(int(q)) or abs(calc_q * float(r) * disc_factor - float(a)) < 0.01):
                                h_item["quantity"] = int(calc_q)
                                h_item["total_quantity"] = int(calc_q) + fq
                                healing_log.append(f"Item #{idx+1}: Healed misread quantity {q} -> {int(calc_q)} based on rate {r} and amount {a}")

            # Item tax amounts derivation
            taxable = h_item.get("taxable_value") or h_item.get("amount")
            cgst_r = h_item.get("cgst_rate")
            sgst_r = h_item.get("sgst_rate")
            igst_r = h_item.get("igst_rate")

            if taxable is not None:
                try:
                    t_val = float(taxable)
                    if (h_item.get("cgst_amount") is None) and cgst_r is not None:
                        h_item["cgst_amount"] = round(t_val * float(cgst_r) / 100.0, 2)
                    if (h_item.get("sgst_amount") is None) and sgst_r is not None:
                        h_item["sgst_amount"] = round(t_val * float(sgst_r) / 100.0, 2)
                    if (h_item.get("igst_amount") is None) and igst_r is not None:
                        h_item["igst_amount"] = round(t_val * float(igst_r) / 100.0, 2)
                except (ValueError, TypeError):
                    pass

            healed_items.append(h_item)

        # ---------------------------------------------------------------------
        # 2. Financial Summary & Tax Rollups
        # ---------------------------------------------------------------------
        line_cgst_sum = sum(float(i["cgst_amount"]) for i in healed_items if i.get("cgst_amount") is not None)
        line_sgst_sum = sum(float(i["sgst_amount"]) for i in healed_items if i.get("sgst_amount") is not None)
        line_igst_sum = sum(float(i["igst_amount"]) for i in healed_items if i.get("igst_amount") is not None)
        line_subtotal_sum = sum(float(i["amount"]) for i in healed_items if i.get("amount") is not None)

        cgst_tot = fin_repaired.get("cgst")
        if (cgst_tot is None or float(cgst_tot) == 0.0) and line_cgst_sum > 0:
            fin_repaired["cgst"] = line_cgst_sum
            healing_log.append(f"Auto-rolled up total CGST amount: {line_cgst_sum}")

        sgst_tot = fin_repaired.get("sgst")
        if (sgst_tot is None or float(sgst_tot) == 0.0) and line_sgst_sum > 0:
            fin_repaired["sgst"] = line_sgst_sum
            healing_log.append(f"Auto-rolled up total SGST amount: {line_sgst_sum}")

        igst_tot = fin_repaired.get("igst")
        if (igst_tot is None or float(igst_tot) == 0.0) and line_igst_sum > 0:
            fin_repaired["igst"] = line_igst_sum
            healing_log.append(f"Auto-rolled up total IGST amount: {line_igst_sum}")

        # Subtotal reconciliation
        ext_subtotal = fin_repaired.get("subtotal")
        subtotal_base = ext_subtotal if ext_subtotal is not None else line_subtotal_sum
        fin_repaired["subtotal"] = subtotal_base

        # Grand Total Auto-Deduction
        taxes_sum = round((fin_repaired.get("cgst") or 0.0) + (fin_repaired.get("sgst") or 0.0) + (fin_repaired.get("igst") or 0.0), 2)
        roundoff = fin_repaired.get("roundoff") or 0.0
        calculated_grand_total = round(subtotal_base + taxes_sum + roundoff, 2)

        ext_grand_total = fin_repaired.get("grand_total")
        if ext_grand_total is None or float(ext_grand_total) == 0.0:
            fin_repaired["grand_total"] = calculated_grand_total
            healing_log.append(f"Auto-deduced missing Grand Total: {calculated_grand_total} (Subtotal {subtotal_base} + Taxes {taxes_sum} + R.Off {roundoff})")
        else:
            # Check if extracted grand total is actually taxable subtotal
            if taxes_sum > 0 and abs(float(ext_grand_total) - subtotal_base) <= 1.0:
                fin_repaired["grand_total"] = calculated_grand_total
                healing_log.append(f"Reconciled mislabeled Grand Total: changed {ext_grand_total} (was subtotal) to {calculated_grand_total}")

        # ---------------------------------------------------------------------
        # 3. Due Date Inference from Credit Terms
        # ---------------------------------------------------------------------
        if not meta_repaired.get("due_date") and all_ocr_lines:
            inv_date_str = meta_repaired.get("invoice_date")
            if inv_date_str:
                parsed_inv_date = cls._parse_date(inv_date_str)
                if parsed_inv_date:
                    # Scan text lines for credit days terms
                    for line in all_ocr_lines:
                        txt = line.get("text", "")
                        credit_days = parse_credit_terms_days(txt)
                        if credit_days:
                            due_dt = parsed_inv_date + timedelta(days=credit_days)
                            meta_repaired["due_date"] = due_dt.strftime("%d/%m/%Y")
                            healing_log.append(f"Inferred due date '{meta_repaired['due_date']}' from credit terms: {credit_days} days")
                            break

        # ---------------------------------------------------------------------
        # 4. GSTIN Recovery from Unassigned Tokens
        # ---------------------------------------------------------------------
        if (not buyer.get("gstin") or not vendor.get("gstin")) and all_ocr_lines:
            gstin_candidates = []
            for line in all_ocr_lines:
                txt = line.get("text", "")
                m = re.search(r"([0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z2][0-9A-Z]{1})", txt)
                if m:
                    cand = m.group(1).upper()
                    # Check if valid state code
                    if cand[:2].isdigit() and 1 <= int(cand[:2]) <= 38:
                        bbox = line.get("bbox", [])
                        gstin_candidates.append({"gstin": cand, "bbox": bbox})

            if gstin_candidates:
                assigned_gstins = {buyer.get("gstin"), vendor.get("gstin")}
                unassigned = [c for c in gstin_candidates if c["gstin"] not in assigned_gstins]

                all_xs = [pt[0] for l in all_ocr_lines for pt in (l.get("bbox") or l.get("box") or [])] if all_ocr_lines else []
                page_width = max(all_xs) if all_xs else 1000.0
                mid_x = page_width * 0.5
                all_ys = [pt[1] for l in all_ocr_lines for pt in (l.get("bbox") or l.get("box") or [])] if all_ocr_lines else []
                page_height = max(all_ys) if all_ys else 1000.0
                header_y_limit = page_height * 0.35

                for cand in unassigned:
                    g = cand["gstin"]
                    c_box = cand.get("bbox") or cand.get("box") or []
                    ymin = min(p[1] for p in c_box) if c_box else 0
                    xmin = min(p[0] for p in c_box) if c_box else 0

                    if not buyer.get("gstin") and (xmin > mid_x or ymin > header_y_limit):
                        buyer["gstin"] = g
                        healing_log.append(f"Recovered customer GSTIN '{g}' from text tokens")
                    elif not vendor.get("gstin") and xmin <= mid_x:
                        vendor["gstin"] = g
                        healing_log.append(f"Recovered seller GSTIN '{g}' from text tokens")

        # Ensure seller and buyer do not share the same GSTIN
        if vendor.get("gstin") and buyer.get("gstin") and vendor["gstin"] == buyer["gstin"]:
            if buyer.get("name") and not vendor.get("name"):
                vendor["gstin"] = None
            else:
                buyer["gstin"] = None
            healing_log.append("Cleared duplicate seller/buyer GSTIN collision")

        meta_repaired["vendor"] = vendor
        meta_repaired["buyer"] = buyer

        return {
            "meta": meta_repaired,
            "fin": fin_repaired,
            "items": healed_items,
            "healing_log": healing_log
        }

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Attempts to parse common date formats into a datetime object."""
        if not date_str:
            return None
        cleaned = date_str.strip().replace("-", "/")
        formats = ["%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None
