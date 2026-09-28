import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from app.ocr.dual_engine_benchmarker import DualEngineBenchmarker
from app.ocr.table_extractor import TableExtractor
from app.ocr.verifier import InvoiceVerifier
from app.ocr.output_formatter import format_official_output
from app.ocr.semantic_ontology import GLOBAL_PARTY_NAME_BLACKLIST
from app.preprocessing.copy_selector import DualCopySelector


LABEL_BLACKLIST = {
    "invoice no", "invoice no.", "inv no", "inv no.", "bill no", "bill no.",
    "invoice date", "dated", "date", "l.r. no", "l.r. no.", "lr no", "lr no.",
    "l.r. date", "lr date", "order no", "order no.", "cases", "due date",
    "gst", "total", "customer", "customer name", "tax invoice", "gst invoice",
    "d.l. no", "fssai no", "pan no", "e-mail", "phone", "state", "details of receiver",
    "buyer (bill to)", "billed to", "bill to", "consignee", "ship to", "buyer",
    "delivery note", "dispatch doc no", "dispatch doc no.", "terms of delivery",
    "mode/terms of payment", "customer copy", "office copy", "credit", "cash", "cash/credit",
    "for orders", "for orders and account", "orders and account"
}

BUYER_ANCHORS = [
    "customer name", "details of receiver", "buyer (bill to)", "billed to",
    "bill to", "consignee", "ship to", "buyer"
]

TABLE_HEADER_KEYWORDS = {
    "batch", "exp", "expiry", "mrp", "rate", "amount", "qty", "quantity", 
    "hsn", "pack", "pcode", "item", "product", "description", "disc", "dis%",
    "free", "mfg", "total", "particulars"
}


class InvoicePipeline:
    """
    End-to-End Invoice Processing Pipeline:
    1. Primary OCR extraction (PaddleOCR on GPU)
    2. Secondary auditing OCR (Tesseract on CPU)
    3. Metadata & Entity Extraction (Vendor, Buyer, Dates, GSTIN, PAN, Bank)
    4. Table & Line-Item Extraction (TableExtractor)
    5. Mathematical & Consensus Verification (InvoiceVerifier)
    """

    def __init__(self, use_gpu: bool = True):
        self.benchmarker = DualEngineBenchmarker(use_gpu=use_gpu)
        self.table_extractor = TableExtractor()
        self.verifier = InvoiceVerifier()
        self.copy_selector = DualCopySelector()

    @staticmethod
    def _extract_invoice_number(extracted_lines: List[Dict[str, Any]]) -> Optional[str]:
        """Extracts primary invoice number using text patterns and spatial box proximity."""
        stopwords = {"has", "is", "the", "date", "dated", "for", "and", "by", "to", "of", "with", "challan"}
        
        # Pass 1: Element contains both label and value (e.g. 'Inv NoC/3930', 'Invoice No: BMS/26-27/196', 'Invoice No.: CC-1415')
        for el in extracted_lines:
            t = el["text"].strip()
            m = re.search(r"\b(?:Tax\s*Invoice|Invoice|Inv|Bill)\s*(?:No\.?|Number|#|Code)[:\s.-]*([A-Za-z0-9][A-Za-z0-9\/\-]+)", t, re.IGNORECASE)
            if m:
                cand = m.group(1).strip(" :-.,")
                cand_lower = cand.lower()
                if (len(cand) >= 3 and cand_lower not in stopwords and 
                    cand_lower not in LABEL_BLACKLIST and 
                    cand_lower not in ["oice", "invoice", "tax invoice", "bill", "tax"] and
                    not cand_lower.startswith(("date", "gst", "total", "customer", "order", "original", "triplicate", "duplicate"))):
                    return cand

        # Pass 2: Spatial Key-Value Search around label 'Invoice No.'
        for el in extracted_lines:
            t_norm = re.sub(r"[.\s]+", " ", el["text"].strip().lower()).strip()
            if t_norm in ["invoice no", "inv no", "bill no", "invoice number"]:
                box = el["box"]
                ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                y_center = (ymin + ymax) / 2.0

                candidates = []
                for other in extracted_lines:
                    if other is el:
                        continue
                    o_text = other["text"].strip()
                    o_clean = o_text.strip(" :-.,")
                    o_lower = o_clean.lower()
                    o_norm = re.sub(r"[.\s]+", " ", o_lower).strip()

                    if (o_lower in LABEL_BLACKLIST or 
                        o_norm in ["invoice no", "inv no", "bill no", "invoice number", "date", "inv date", "dated", "tax invoice"] or
                        any(o_lower.startswith(k) for k in ["invoice", "date", "gst", "total", "customer", "order", "state", "pan"])):
                        continue
                    if len(o_clean) < 3 or re.match(r"^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}$", o_clean):
                        continue

                    o_box = other["box"]
                    o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                    o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                    o_ycenter = (o_ymin + o_ymax) / 2.0

                    # Check horizontally to the right on the same row (within 25px Y)
                    if abs(o_ycenter - y_center) <= 25.0 and 0 < (o_xmin - xmax) <= 400:
                        candidates.append((o_xmin - xmax, o_clean, "right"))
                    # Check vertically directly below (within 60px Y, overlapping X)
                    elif 0 < (o_ymin - ymax) <= 60 and not (o_xmax < xmin or o_xmin > xmax):
                        candidates.append((o_ymin - ymax, o_clean, "below"))

                if candidates:
                    candidates.sort(key=lambda c: (0 if c[2] == "right" else 1, c[0]))
                    return candidates[0][1]

        return None

    @staticmethod
    def _extract_invoice_date(extracted_lines: List[Dict[str, Any]]) -> Optional[str]:
        """Extracts invoice date using strict priority hierarchy and spatial proximity."""
        DATE_REGEX = r"([0-9]{1,2}[-\/.](?:[0-9]{1,2}|[A-Za-z]{3})[-\/.][0-9]{2,4})"
        NON_INV_KEYWORDS = r"\b(?:Due|DueDt|Order|PO|Challan|DC|LR|Mfg|Exp|Delivery)\b|dDate"

        # Priority 1: Explicit labeled pattern in the same element (e.g. 'Inv Date: 26/06/2026', 'Invoice Date: 03/08/2026')
        for el in extracted_lines:
            t = el["text"].strip()
            if re.search(NON_INV_KEYWORDS, t, re.IGNORECASE):
                continue
            m = re.search(r"(?:Invoice\s*(?:Date|Dt\.?)|Inv\s*(?:Date|Dt\.?)|InvDt|Bill\s*(?:Date|Dt\.?))[:\s.-]*" + DATE_REGEX, t, re.IGNORECASE)
            if m:
                return m.group(1).replace(".", "/")

        # Priority 2: Explicit invoice date label with horizontal or vertical spatial neighbor
        for el in extracted_lines:
            t = el["text"].strip().lower()
            if t in ["invoice date", "inv date", "inv dt", "invdt", "bill date", "bill dt"]:
                box = el["box"]
                ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                y_center = (ymin + ymax) / 2.0

                for other in extracted_lines:
                    if other is el:
                        continue
                    m = re.search(DATE_REGEX, other["text"])
                    if not m:
                        continue
                    o_box = other["box"]
                    o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                    o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                    o_ycenter = (o_ymin + o_ymax) / 2.0

                    # Check horizontally to the right on the same row (within 25px Y)
                    if abs(o_ycenter - y_center) <= 25 and 0 < (o_xmin - xmax) <= 350:
                        return m.group(1).replace(".", "/")
                    # Check vertically directly below
                    if 0 < (o_ymin - ymin) < 60 and not (o_xmax < xmin or o_xmin > xmax):
                        return m.group(1).replace(".", "/")

        # Priority 3: Generic 'Date:' or 'Dated:' (ensuring it's not Order/PO/Delivery date)
        for el in extracted_lines:
            t = el["text"].strip()
            if re.search(NON_INV_KEYWORDS, t, re.IGNORECASE):
                continue
            m = re.search(r"(?:Dated|Date)[:\s.-]*" + DATE_REGEX, t, re.IGNORECASE)
            if m:
                return m.group(1).replace(".", "/")

        # Priority 4: Generic date label with spatial neighbor
        for el in extracted_lines:
            t = el["text"].strip().lower()
            if t in ["dated", "date"]:
                box = el["box"]
                ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                y_center = (ymin + ymax) / 2.0

                for other in extracted_lines:
                    if other is el:
                        continue
                    m = re.search(DATE_REGEX, other["text"])
                    if not m:
                        continue
                    o_box = other["box"]
                    o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                    o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                    o_ycenter = (o_ymin + o_ymax) / 2.0

                    if abs(o_ycenter - y_center) <= 25 and 0 < (o_xmin - xmax) <= 300:
                        return m.group(1).replace(".", "/")
                    elif 0 < (o_ymin - ymin) < 60 and not (o_xmax < xmin or o_xmin > xmax):
                        return m.group(1).replace(".", "/")

        # Priority 5: Fallback regex across header elements (Y < 400)
        for el in extracted_lines:
            if min(p[1] for p in el["box"]) < 400:
                t = el["text"]
                if not re.search(NON_INV_KEYWORDS, t, re.IGNORECASE):
                    m = re.search(DATE_REGEX, t)
                    if m:
                        return m.group(1).replace(".", "/")

        return None

    @staticmethod
    def _extract_po_number(extracted_lines: List[Dict[str, Any]]) -> Optional[str]:
        """Extracts purchase order / PO number using pattern matching and spatial proximity."""
        def _clean_po(val: str) -> str:
            cleaned = val.strip(" :-.,")
            # Strip redundant prefix like PONO. or PO NO.
            cleaned = re.sub(r"^(?:PO\s*NO\.?|P\.?O\.?\s*NO\.?|NO\.?)\s*", "", cleaned, flags=re.IGNORECASE)
            # Strip trailing tags like Or or Dt
            cleaned = re.sub(r"\s*(?:Or|Dt|Date)$", "", cleaned, flags=re.IGNORECASE)
            # Restore slash between financial year and serial number if concatenated by OCR, e.g. 2026-277326 -> 2026-27/7326
            cleaned = re.sub(r"(\b\d{2,4}-\d{2})(\d{3,})", r"\1/\2", cleaned)
            return cleaned.strip(" :-.,")

        # Pass 1: Explicit PO code pattern across all lines (e.g. DMH/PO/PHRMCY/2026-27/426)
        for el in extracted_lines:
            t = el["text"].strip()
            m = re.search(r"\b([A-Za-z0-9_\/-]*\/?PO\/[A-Za-z0-9_\/-]+)\b", t)
            if m:
                cand = _clean_po(m.group(1))
                if len(cand) >= 5:
                    return cand

        # Pass 2: Label + value on the same line
        for el in extracted_lines:
            t = el["text"].strip()
            m = re.search(r"(?:P\.?O\.?\s*(?:No\.?|Number|#)?|Purchase\s*Order(?:\s*No\.?)?|Order\s*No\.?)[:\s.-]+([A-Za-z0-9\/\-_.]{3,})", t, re.IGNORECASE)
            if m:
                cand = _clean_po(m.group(1))
                if len(cand) >= 3 and not cand.lower().startswith(("date", "gst", "total")):
                    return cand

        # Pass 3: Spatial proximity around 'P.O. No' or 'Order No'
        for el in extracted_lines:
            t = el["text"].strip().lower().rstrip(" :.-")
            if t in ["p.o. no", "po no", "p.o. no.", "order no", "order no.", "po number"]:
                box = el["box"]
                ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                y_center = (ymin + ymax) / 2.0
                candidates = []
                for other in extracted_lines:
                    if other is el:
                        continue
                    o_clean = _clean_po(other["text"])
                    o_lower = o_clean.lower()
                    if o_lower in LABEL_BLACKLIST or len(o_clean) < 3:
                        continue
                    o_box = other["box"]
                    o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                    o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                    o_ycenter = (o_ymin + o_ymax) / 2.0
                    if abs(o_ycenter - y_center) <= 25.0 and 0 < (o_xmin - xmax) <= 400:
                        candidates.append((o_xmin - xmax, o_clean, "right"))
                    elif 0 < (o_ymin - ymax) <= 60 and not (o_xmax < xmin or o_xmin > xmax):
                        candidates.append((o_ymin - ymax, o_clean, "below"))
                if candidates:
                    candidates.sort(key=lambda c: (0 if c[2] == "right" else 1, c[0]))
                    return candidates[0][1]

        return None

    @staticmethod
    def _extract_due_date(extracted_lines: List[Dict[str, Any]]) -> Optional[str]:
        """Extracts due date using label pattern matching and spatial proximity."""
        # Pass 1: Label + value on the same line
        for el in extracted_lines:
            t = el["text"].strip()
            m = re.search(r"(?:Due\s*(?:Date|Dt\.?))[:\s.-]*([0-9]{1,2}[-\/.](?:[0-9]{1,2}|[A-Za-z]{3})[-\/.][0-9]{2,4})", t, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Pass 2: Spatial search near 'Due Date' / 'Due Dt' label
        for el in extracted_lines:
            t = el["text"].strip().lower()
            if any(k in t for k in ["due date", "due dt"]):
                box = el["box"]
                ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                y_center = (ymin + ymax) / 2.0
                candidates = []
                for other in extracted_lines:
                    if other is el:
                        continue
                    o_text = other["text"].strip()
                    m_d = re.search(r"\b([0-9]{1,2}[-\/.](?:[0-9]{1,2}|[A-Za-z]{3})[-\/.][0-9]{2,4})\b", o_text)
                    if not m_d:
                        continue
                    o_box = other["box"]
                    o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                    o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                    o_ycenter = (o_ymin + o_ymax) / 2.0
                    if abs(o_ycenter - y_center) <= 25.0 and 0 < (o_xmin - xmax) <= 400:
                        candidates.append((o_xmin - xmax, m_d.group(1), "right"))
                    elif 0 < (o_ymin - ymax) <= 60 and not (o_xmax < xmin or o_xmin > xmax):
                        candidates.append((o_ymin - ymax, m_d.group(1), "below"))
                if candidates:
                    candidates.sort(key=lambda c: (0 if c[2] == "right" else 1, c[0]))
                    return candidates[0][1]

        return None

    @staticmethod
    def _extract_taxes_and_totals(
        text: str,
        extracted_lines: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Optional[float]]:
        """Extracts financial summary: Subtotal, CGST, SGST, IGST, Roundoff, Grand Total."""
        result = {
            "subtotal": None,
            "cgst": None,
            "sgst": None,
            "igst": None,
            "roundoff": None,
            "grand_total": None
        }

        # CGST
        cgst_m = re.findall(r"(?:CGST|CGST\s*PAYBLE|Out\s*Put\s*CGST)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if cgst_m:
            for val_s in reversed(cgst_m):
                try:
                    v = float(val_s.replace(",", ""))
                    if v > 0:
                        result["cgst"] = v
                        break
                except ValueError:
                    pass

        # SGST
        sgst_m = re.findall(r"(?:SGST|SGST\s*PAYBLE|Out\s*Put\s*SGST)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if sgst_m:
            for val_s in reversed(sgst_m):
                try:
                    v = float(val_s.replace(",", ""))
                    if v > 0:
                        result["sgst"] = v
                        break
                except ValueError:
                    pass

        # IGST
        igst_m = re.findall(r"(?:IGST|Integrated\s*Tax)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if igst_m:
            for val_s in reversed(igst_m):
                try:
                    v = float(val_s.replace(",", ""))
                    if v > 0:
                        result["igst"] = v
                        break
                except ValueError:
                    pass

        # Round off
        round_m = re.findall(r"(?:Round\s*off)[\s:₹Rs\.]*([0-9\.\-]+)", text, re.IGNORECASE)
        if round_m:
            try:
                result["roundoff"] = float(round_m[-1].replace(",", ""))
            except ValueError:
                pass

        if result["roundoff"] is None and extracted_lines:
            for el in extracted_lines:
                t = el["text"].strip().lower()
                if "round" in t and "off" in t:
                    box = el["box"]
                    ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                    xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                    y_center = (ymin + ymax) / 2.0
                    for other in extracted_lines:
                        if other is el:
                            continue
                        o_t = other["text"].strip().replace(",", "")
                        m_ro = re.match(r"^(-?[0-9]*\.[0-9]{1,2})$", o_t)
                        if m_ro:
                            o_box = other["box"]
                            o_ycenter = (min(p[1] for p in o_box) + max(p[1] for p in o_box)) / 2.0
                            o_xmin = min(p[0] for p in o_box)
                            if abs(o_ycenter - y_center) <= 35.0 and o_xmin > xmin:
                                try:
                                    val_str = m_ro.group(1)
                                    if val_str.startswith("-."):
                                        val_str = "-0." + val_str[2:]
                                    elif val_str.startswith("."):
                                        val_str = "0." + val_str[1:]
                                    result["roundoff"] = float(val_str)
                                    break
                                except ValueError:
                                    pass
                    if result["roundoff"] is not None:
                        break

        # Grand Total:
        # Pass 1: Specific labeled patterns on same line
        total_m = re.findall(r"(?:Grand\s*Total|Total\s*Amount|Amount\s*Payable|Net\s*Amt\.?|Net\s*Amount|TO\s*PAY|TOPAY|Inv\s*Amt\.?|Invoice\s*Amt\.?|Total\s*Amt\.?)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if total_m:
            try:
                result["grand_total"] = float(total_m[-1].replace(",", ""))
            except ValueError:
                pass

        # Pass 1B: Trailing TO PAY pattern (e.g. '56622.28TOPAY')
        if not result["grand_total"]:
            m_topay = re.findall(r"([0-9,]+\.[0-9]{2})\s*(?:TOPAY|TO\s*PAY)", text, re.IGNORECASE)
            if m_topay:
                try:
                    result["grand_total"] = float(m_topay[-1].replace(",", ""))
                except ValueError:
                    pass

        # Pass 2: Spatial search for Grand Total label if lines available
        if not result["grand_total"] and extracted_lines:
            for el in extracted_lines:
                t = el["text"].strip().lower()
                if t in ["grand total", "total amount", "amount payable", "net amount", "net amt", "to pay", "topay", "inv amt", "inv amt :", "total amt"]:
                    box = el["box"]
                    ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
                    xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
                    y_center = (ymin + ymax) / 2.0

                    candidates = []
                    for other in extracted_lines:
                        if other is el:
                            continue
                        o_text = other["text"].strip().replace(",", "")
                        if not re.match(r"^[0-9]+\.[0-9]{2}$", o_text):
                            continue
                        val = float(o_text)
                        if val <= 10.0:  # Skip trivial numbers
                            continue

                        o_box = other["box"]
                        o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                        o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                        o_ycenter = (o_ymin + o_ymax) / 2.0

                        # Same row to the right
                        if abs(o_ycenter - y_center) <= 30.0 and 0 < (o_xmin - xmax) <= 500:
                            candidates.append((o_xmin - xmax, val, "right"))
                        # Below
                        elif 0 < (o_ymin - ymax) <= 100 and abs(o_xmin - xmin) <= 150:
                            candidates.append((o_ymin - ymax, val, "below"))

                    if candidates:
                        candidates.sort(key=lambda c: (0 if c[2] == "right" else 1, c[0]))
                        result["grand_total"] = candidates[0][1]
                        break

        # Pass 3: Fallback to plain Total if no grand total pattern matched
        if not result["grand_total"]:
            total_fallback = re.findall(r"(?:Total)[\s:₹Rs\.]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
            if total_fallback:
                try:
                    result["grand_total"] = float(total_fallback[-1].replace(",", ""))
                except ValueError:
                    pass

        # Spatial column search for CGST/SGST if not matched by regex
        if (result["cgst"] is None or result["sgst"] is None) and extracted_lines:
            for el in extracted_lines:
                t_lower = el["text"].strip().lower()
                b = el["box"]
                xmin, xmax = min(p[0] for p in b), max(p[0] for p in b)
                ymin, ymax = min(p[1] for p in b), max(p[1] for p in b)

                # Check if this element contains both CGST and SGST headers in one string (e.g. 'CGSTCGST AMTSGSTSGST AMT')
                if "cgst" in t_lower and "sgst" in t_lower:
                    w = xmax - xmin
                    has_taxable = "taxable" in t_lower
                    has_total_col = "total" in t_lower
                    if has_taxable:
                        # 3 primary sections: taxable (0-30%), cgst_amt (35-65%), sgst_amt (70-100%)
                        q2_min, q2_max = xmin + 0.35 * w, xmin + 0.65 * w
                        q4_min, q4_max = xmin + 0.70 * w, xmax + 25
                    elif has_total_col:
                        # 5 horizontal sections:
                        # cgst_rate (0-15%), cgst_amt (15-38%), sgst_rate (38-52%), sgst_amt (52-77%), total_amt (77-100%)
                        q2_min, q2_max = xmin + 0.15 * w, xmin + 0.38 * w
                        q4_min, q4_max = xmin + 0.52 * w, xmin + 0.77 * w
                    else:
                        # 4 horizontal sections: cgst_rate (0-22%), cgst_amt (22-52%), sgst_rate (52-72%), sgst_amt (72-100%)
                        q2_min, q2_max = xmin + 0.22 * w, xmin + 0.52 * w
                        q4_min, q4_max = xmin + 0.72 * w, xmax + 20

                    for other in extracted_lines:
                        if other is el:
                            continue
                        o_b = other["box"]
                        o_xmin, o_xmax = min(p[0] for p in o_b), max(p[0] for p in o_b)
                        o_ymin = min(p[1] for p in o_b)
                        if 0 < (o_ymin - ymax) <= 80:
                            m = re.match(r"^([0-9,]+\.[0-9]{2})$", other["text"].strip())
                            if m:
                                val = float(m.group(1).replace(",", ""))
                                if has_taxable and o_xmin < xmin + 0.30 * w and not result.get("taxable_amount"):
                                    result["taxable_amount"] = val
                                if result["cgst"] is None and not (o_xmax < q2_min or o_xmin > q2_max):
                                    result["cgst"] = val
                                elif result["sgst"] is None and not (o_xmax < q4_min or o_xmin > q4_max):
                                    result["sgst"] = val
                else:
                    if "sgst amt" in t_lower and result["sgst"] is None:
                        col_vals = []
                        for other in extracted_lines:
                            if other is el:
                                continue
                            o_b = other["box"]
                            o_xmin, o_xmax = min(p[0] for p in o_b), max(p[0] for p in o_b)
                            o_ymin = min(p[1] for p in o_b)
                            if 0 < (o_ymin - ymax) <= 80 and not (o_xmax < xmin - 30 or o_xmin > xmax + 30):
                                m = re.match(r"^([0-9,]+\.[0-9]{2})$", other["text"].strip())
                                if m:
                                    col_vals.append((o_ymin, float(m.group(1).replace(",", ""))))
                        if col_vals:
                            col_vals.sort(key=lambda c: c[0])
                            result["sgst"] = col_vals[0][1]

                    if "cgst amt" in t_lower and result["cgst"] is None:
                        col_vals = []
                        for other in extracted_lines:
                            if other is el:
                                continue
                            o_b = other["box"]
                            o_xmin, o_xmax = min(p[0] for p in o_b), max(p[0] for p in o_b)
                            o_ymin = min(p[1] for p in o_b)
                            if 0 < (o_ymin - ymax) <= 80 and not (o_xmax < xmin - 30 or o_xmin > xmax + 30):
                                m = re.match(r"^([0-9,]+\.[0-9]{2})$", other["text"].strip())
                                if m:
                                    col_vals.append((o_ymin, float(m.group(1).replace(",", ""))))
                        if col_vals:
                            col_vals.sort(key=lambda c: c[0])
                            result["cgst"] = col_vals[0][1]

        # Additional financial breakdown fields
        disc_m = re.findall(r"(?:DIS\s*AMT\.?|DISCOUNT|TOTAL\s*DISCOUNT)[\s:]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if disc_m:
            try:
                result["discount_amount"] = float(disc_m[-1].replace(",", ""))
            except ValueError:
                pass

        tgst_m = re.findall(r"TOTAL\s*GST[\s:]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if tgst_m:
            try:
                result["total_gst"] = float(tgst_m[-1].replace(",", ""))
            except ValueError:
                pass
        elif result.get("cgst") and result.get("sgst"):
            result["total_gst"] = round(result["cgst"] + result["sgst"] + (result.get("igst") or 0.0), 2)

        taxable_m = re.findall(r"(?:TAXABLE\s*(?:VALUE|AMT\.?|AMOUNT))[\s:]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if taxable_m:
            try:
                result["taxable_amount"] = float(taxable_m[-1].replace(",", ""))
            except ValueError:
                pass

        # Spatial search for Taxable Amount if extracted_lines available
        if extracted_lines:
            taxable_candidates = []
            for el in extracted_lines:
                txt = el.get("text", "").strip()
                if re.fullmatch(r"Taxable\s*(?:Amt\.?|Amount|Value)", txt, re.IGNORECASE):
                    box = el.get("box")
                    if box:
                        lbl_y = (box[0][1] + box[2][1]) / 2.0
                        lbl_x_max = max(pt[0] for pt in box)
                        for cand in extracted_lines:
                            cand_box = cand.get("box")
                            if not cand_box:
                                continue
                            cand_y = (cand_box[0][1] + cand_box[2][1]) / 2.0
                            cand_x_min = min(pt[0] for pt in cand_box)
                            if abs(cand_y - lbl_y) < 15 and 0 <= (cand_x_min - lbl_x_max) < 350:
                                num_m = re.search(r"([0-9,]+\.[0-9]{2})", cand.get("text", ""))
                                if num_m:
                                    try:
                                        c_val = float(num_m.group(1).replace(",", ""))
                                        if c_val > 0:
                                            taxable_candidates.append(c_val)
                                    except ValueError:
                                        pass
            if taxable_candidates:
                gt = result.get("grand_total")
                tg = result.get("total_gst")
                matched = False
                if gt and tg:
                    for tc in taxable_candidates:
                        if abs(gt - (tc + tg)) < 2.0:
                            result["taxable_amount"] = tc
                            matched = True
                            break
                if not matched:
                    result["taxable_amount"] = max(taxable_candidates)

        # Derive Round off if missing
        if result["roundoff"] is None and result["grand_total"] and result.get("taxable_amount") and result.get("total_gst"):
            diff = round(result["grand_total"] - (result["taxable_amount"] + result["total_gst"]), 2)
            if abs(diff) < 2.0:
                result["roundoff"] = diff

        qty_m = re.search(r"Total\s*Qty\s*[:-]+[\s]*([0-9]+)", text, re.IGNORECASE)
        if qty_m:
            try:
                result["total_quantity"] = int(qty_m.group(1))
            except ValueError:
                pass

        items_m = re.search(r"Total\s*Items\s*[:-]+[\s]*([0-9]+)", text, re.IGNORECASE)
        if items_m:
            try:
                result["total_items"] = int(items_m.group(1))
            except ValueError:
                pass

        cr_dr_m = re.search(r"CR\/DR\s*NOTE[\s:]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if cr_dr_m:
            try:
                result["cr_dr_note"] = float(cr_dr_m.group(1).replace(",", ""))
            except ValueError:
                pass
        else:
            result["cr_dr_note"] = 0.0

        tcs_m = re.search(r"TCS\s*TAX[\s\S]{0,30}?([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if tcs_m:
            try:
                result["tcs_tax"] = float(tcs_m.group(1).replace(",", ""))
            except ValueError:
                pass
        else:
            result["tcs_tax"] = 0.0

        courier_m = re.search(r"Courier\s*(?:Expanses|Expenses)[\s:]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
        if courier_m:
            try:
                result["courier_expenses"] = float(courier_m.group(1).replace(",", ""))
            except ValueError:
                pass
        else:
            result["courier_expenses"] = 0.0

        return result

    @staticmethod
    def _extract_parties_metadata(extracted_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts vendor and buyer information based on spatial column positions and keywords."""
        # Consider header elements above table rows (Y < 620)
        header_elements = [el for el in extracted_lines if min(p[1] for p in el["box"]) < 620]

        buyer_anchor_el = None
        for el in header_elements:
            t_lower = el["text"].strip().lower().rstrip(" :.-")
            if any(bad in t_lower for bad in ["order", "po ", "po.", "challan", "date", "no.", "no:"]):
                continue
            for anchor in BUYER_ANCHORS:
                if anchor in t_lower:
                    buyer_anchor_el = el
                    break
            if buyer_anchor_el:
                break

        buyer_name = None
        buyer_box = None

        if buyer_anchor_el:
            b_box = buyer_anchor_el["box"]
            b_ymin, b_ymax = min(p[1] for p in b_box), max(p[1] for p in b_box)
            b_xmin, b_xmax = min(p[0] for p in b_box), max(p[0] for p in b_box)
            b_ycenter = (b_ymin + b_ymax) / 2.0

            buyer_box = (b_xmin - 50, b_xmax + 350)

            candidates = []
            for el in header_elements:
                if el is buyer_anchor_el:
                    continue
                t = el["text"].strip()
                t_clean = t.rstrip(" :.-")
                t_lower = t_clean.lower()
                if (t_lower in LABEL_BLACKLIST or 
                    any(t_lower.startswith(k) for k in ["gst", "pan", "state", "address", "phone", "email", "dl", "d.l", "inv", "date"])):
                    continue
                if len(t) < 3:
                    continue

                o_box = el["box"]
                o_ymin, o_ymax = min(p[1] for p in o_box), max(p[1] for p in o_box)
                o_xmin, o_xmax = min(p[0] for p in o_box), max(p[0] for p in o_box)
                o_ycenter = (o_ymin + o_ymax) / 2.0

                # Directly below (within 60px)
                if 0 < (o_ymin - b_ymin) <= 60 and (abs(o_xmin - b_xmin) < 100 or not (o_xmax < b_xmin or o_xmin > b_xmax)):
                    candidates.append((o_ymin - b_ymin, t, "below"))
                # Right next to on same line
                elif abs(o_ycenter - b_ycenter) <= 20 and 0 < (o_xmin - b_xmax) <= 300:
                    candidates.append((o_xmin - b_xmax, t, "right"))

            if candidates:
                candidates.sort(key=lambda c: (0 if c[2] == "right" else 1, c[0]))
                buyer_name = candidates[0][1]

        # Labeled buyer row fallback (e.g. 'Name : ACME MEDICAL STORES' on same line or next line)
        if not buyer_name:
            for el in header_elements:
                t = el["text"].strip()
                t_lower = t.lower().rstrip(" :.-")
                # Case A: Same line 'Name : ACME MEDICAL STORES'
                m_same = re.match(r"^(?:Name|Party\s*Name|Customer\s*Name|M\/s)[:\s.-]+([A-Za-z0-9\s.,&'-]{3,})", t, re.IGNORECASE)
                if m_same:
                    cand_name = m_same.group(1).strip(" :.-")
                    if cand_name.lower() not in LABEL_BLACKLIST and len(cand_name) >= 3 and not any(k in cand_name.lower().split() for k in TABLE_HEADER_KEYWORDS):
                        buyer_name = cand_name
                        break

                # Case B: 'Name :' on one element, value on adjacent element
                if t_lower in ["name", "party name", "customer name", "m/s"]:
                    b_box = el["box"]
                    b_xmax = max(p[0] for p in b_box)
                    b_ycenter = (min(p[1] for p in b_box) + max(p[1] for p in b_box)) / 2.0
                    for other in header_elements:
                        if other is el:
                            continue
                        o_box = other["box"]
                        o_xmin = min(p[0] for p in o_box)
                        o_ycenter = (min(p[1] for p in o_box) + max(p[1] for p in o_box)) / 2.0
                        if abs(o_ycenter - b_ycenter) <= 20 and 0 < (o_xmin - b_xmax) <= 450:
                            cand_name = other["text"].strip(" :.-")
                            if len(cand_name) >= 3 and cand_name.lower() not in LABEL_BLACKLIST:
                                buyer_name = cand_name
                                break
                    if buyer_name:
                        break

        # Two-column layout fallback (e.g. Customer in right column 450 <= X <= 2600, Y < 350)
        if not buyer_name:
            col_cands = []
            for el in header_elements:
                b_box = el["box"]
                xmin = min(p[0] for p in b_box)
                ymin = min(p[1] for p in b_box)
                if 450 <= xmin <= 2600 and ymin < 350:
                    t_clean = el["text"].strip(" :.-")
                    t_lower = t_clean.lower()
                    if (t_lower in LABEL_BLACKLIST or 
                        t_lower in GLOBAL_PARTY_NAME_BLACKLIST or
                        any(anchor in t_lower for anchor in BUYER_ANCHORS) or
                        t_lower in TABLE_HEADER_KEYWORDS or
                        any(k in t_lower.split() for k in TABLE_HEADER_KEYWORDS) or
                        any(t_lower.startswith(k) for k in ["tax", "gst", "pan", "invoice", "inv", "customer copy", "original", "page", "phone", "ph", "email", "d.l", "fssai", "ord", "due", "date", "bill"]) or
                        re.search(r"\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4}", t_clean)):
                        continue
                    alpha_words = re.findall(r"[A-Za-z]{2,}", t_clean)
                    if not alpha_words or max(len(w) for w in alpha_words) < 3 or sum(len(w) for w in alpha_words) < 4:
                        continue
                    col_cands.append((ymin, t_clean))
            if col_cands:
                col_cands.sort(key=lambda c: c[0])
                buyer_name = col_cands[0][1]

        if buyer_name and "MEDICALSTORES" in buyer_name:
            buyer_name = buyer_name.replace("MEDICALSTORES", "MEDICAL STORES")

        # Vendor Name Extraction (top section above table)
        vendor_name = None
        for el in [e for e in header_elements if min(p[1] for p in e["box"]) < 500]:
            t = el["text"].strip()
            t_clean = t.rstrip(" :.-")
            t_lower = t_clean.lower()

            if (t_lower in LABEL_BLACKLIST or 
                t_lower in TABLE_HEADER_KEYWORDS or 
                t_lower in GLOBAL_PARTY_NAME_BLACKLIST or
                any(k in t_lower.split() for k in TABLE_HEADER_KEYWORDS) or
                "product description" in t_lower or
                "item name" in t_lower):
                continue
            if any(t_lower.startswith(k) for k in [
                "tax", "gst", "invoice", "original", "page", "phone", "email", "d.l", 
                "fssai", "customer copy", "office copy", "credit", "cash", "shop no", 
                "beside", "near", "opp", "plot no", "flat no", "survey no", "gat no"
            ]):
                continue
            if any(bad in t_lower for bad in [
                "for orders", "for order", "orders and account", "order and account", 
                "contact", "support", "helpdesk", "statement", "disposable", 
                "disinfectant", "wound care", "dressing"
            ]):
                continue
            if min(p[0] for p in el["box"]) > 1800:
                continue
            if re.search(r"\d{7,}", t_clean):
                continue
            if re.match(r"^[A-Za-z](?:,[A-Za-z])+$", t_clean):
                continue
            alpha_words = re.findall(r"[A-Za-z]{2,}", t_clean)
            if not alpha_words or max(len(w) for w in alpha_words) < 3 or sum(len(w) for w in alpha_words) < 4:
                continue

            if buyer_anchor_el:
                el_xmin = min(p[0] for p in el["box"])
                el_ymin = min(p[1] for p in el["box"])
                b_ymin = min(p[1] for p in buyer_anchor_el["box"])
                if el_ymin >= b_ymin - 20:
                    if buyer_box and buyer_box[0] <= el_xmin <= buyer_box[1]:
                        continue
                if min(p[0] for p in buyer_anchor_el["box"]) > 1000 and el_xmin > 1000:
                    continue

            if re.search(r"^\d+\s*days?$", t_clean, re.IGNORECASE):
                continue

            if min(p[0] for p in el["box"]) > 800:
                continue

            if min(p[0] for p in el["box"]) > 450 and buyer_name == t_clean:
                continue

            if re.search(r"\b(?:road|street|lane|marg|nagar|sector|phase|pincode|pin\s*code|dist|tal)\b", t_lower) or re.search(r"\b\d{3}\s*\d{3}\b", t_lower):
                continue

            if len(t) >= 4:
                vendor_name = t
                break

        # Fallback vendor extraction from authorization signature line (e.g. 'For VENDOR TRADING CO')
        if not vendor_name:
            for el in extracted_lines:
                t = el["text"].strip()
                m_for = re.match(r"^For\s+([A-Za-z0-9\s.,&'-]{3,})", t, re.IGNORECASE)
                if m_for:
                    cand = m_for.group(1).strip()
                    cand_clean = cand.lower()
                    if cand_clean not in LABEL_BLACKLIST and cand_clean not in GLOBAL_PARTY_NAME_BLACKLIST and not any(bad in cand_clean for bad in ["orders", "account"]) and len(cand) >= 3:
                        vendor_name = cand
                        break

        # GSTIN Extraction & Spatial Attribution (with 0/O OCR tolerance)
        gstin_elements = []
        gstin_regex = r"([0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z2][0-9A-Z]{1})"
        for el in header_elements:
            m = re.search(gstin_regex, el["text"])
            if m:
                raw_gstin = m.group(1)
                prefix = raw_gstin[:2]
                pan_letters = raw_gstin[2:7].replace("0", "O")
                suffix = raw_gstin[7:]
                gstin_val = prefix + pan_letters + suffix
                el_box = el["box"]
                gstin_elements.append({
                    "gstin": gstin_val,
                    "xmin": min(p[0] for p in el_box),
                    "ymin": min(p[1] for p in el_box),
                    "text": el["text"]
                })

        vendor_gstin = None
        buyer_gstin = None

        if len(gstin_elements) >= 2:
            g_left = [g for g in gstin_elements if g["xmin"] < 500]
            g_right = [g for g in gstin_elements if g["xmin"] >= 500]

            if g_left and g_right:
                # Distinct horizontal columns: left is Vendor, right is Buyer
                vendor_gstin = g_left[0]["gstin"]
                buyer_gstin = g_right[0]["gstin"]
            else:
                # Both in same column: use vertical Y-separation with buyer anchor
                if buyer_anchor_el:
                    anchor_ymin = min(p[1] for p in buyer_anchor_el["box"])
                    for g in gstin_elements:
                        if g["ymin"] < anchor_ymin:
                            vendor_gstin = g["gstin"]
                        else:
                            buyer_gstin = g["gstin"]
                elif buyer_box:
                    for g in gstin_elements:
                        if buyer_box[0] <= g["xmin"] <= buyer_box[1]:
                            buyer_gstin = g["gstin"]
                        else:
                            vendor_gstin = g["gstin"]
                else:
                    sorted_by_y = sorted(gstin_elements, key=lambda g: g["ymin"])
                    vendor_gstin = sorted_by_y[0]["gstin"]
                    buyer_gstin = sorted_by_y[1]["gstin"]
        elif len(gstin_elements) == 1:
            if gstin_elements[0]["xmin"] >= 480:
                buyer_gstin = gstin_elements[0]["gstin"]
            else:
                vendor_gstin = gstin_elements[0]["gstin"]

        full_text_combined = "\n".join(l["text"] for l in extracted_lines)

        # Derive vendor PAN from vendor GSTIN characters 3..12 or regex
        if vendor_gstin and len(vendor_gstin) == 15:
            pan_no = vendor_gstin[2:12]
        else:
            pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b", full_text_combined)
            pan_no = pan_match.group(1) if pan_match else None

        # Bank Details
        bank_name = None
        bnk_m = re.search(r"Bank\s*Name[:\s]*([A-Za-z\s]+)", full_text_combined, re.IGNORECASE)
        if bnk_m:
            raw_bnk = bnk_m.group(1).strip()
            bank_name = re.split(r"Branch\s*Name|A\/c|Account|IFSC|\n", raw_bnk, flags=re.IGNORECASE)[0].strip()

        bank_account = None
        acc_m = re.search(
            r"(?:A\/c\s*No\.?|Account\s*No\.?|A\/C\s*NUMBER|ACCOUNT\s*NUMBER|A\/C\.?\s*NO\.?)[:\s.-]*([0-9]{9,18})",
            full_text_combined,
            re.IGNORECASE
        )
        if acc_m:
            bank_account = acc_m.group(1).strip()

        ifsc_code = None
        ifsc_m = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", full_text_combined)
        if ifsc_m:
            ifsc_code = ifsc_m.group(1)

        return {
            "vendor_name": vendor_name,
            "vendor_gstin": vendor_gstin,
            "buyer_name": buyer_name,
            "buyer_gstin": buyer_gstin,
            "pan": pan_no,
            "vendor_pan": pan_no,
            "bank_details": {
                "bank_name": bank_name,
                "account_number": bank_account,
                "ifsc_code": ifsc_code
            }
        }

    def process_invoice(
        self,
        image_path: Path,
        return_diagnostics: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Processes a single deskewed invoice image end-to-end and outputs official schema."""
        start_pipeline = time.time()
        img_path = Path(image_path)
        if not img_path.exists():
            err_res = {"error": f"Image file {img_path} not found"}
            return (err_res, {}) if return_diagnostics else err_res

        if img_path.suffix.lower() == ".pdf":
            from app.preprocessing.image_converter import convert_file
            conv_dir = Path("data/test_uploads/staging/converted")
            conv_dir.mkdir(parents=True, exist_ok=True)
            conv_paths = convert_file(str(img_path), str(conv_dir))
            if conv_paths:
                img_path = Path(conv_paths[0])

        # 0. Check for dual-copy page layout (Customer Copy vs Office Copy)
        copy_res = self.copy_selector.inspect_and_select(img_path)
        if copy_res.is_dual_copy:
            img_path = copy_res.output_image_path

        # 1. Primary PaddleOCR Extraction
        paddle_res = self.benchmarker.run_paddle(img_path)
        extracted_lines = paddle_res.get("extracted_lines", [])
        paddle_full_text = paddle_res.get("full_text", "")

        # 2. Secondary Tesseract OCR Extraction
        tesseract_res = self.benchmarker.run_tesseract(img_path)
        tesseract_full_text = tesseract_res.get("full_text", "")

        # 3. Metadata Extraction
        parties = self._extract_parties_metadata(extracted_lines)
        inv_number = self._extract_invoice_number(extracted_lines)
        inv_date = self._extract_invoice_date(extracted_lines)
        due_date = self._extract_due_date(extracted_lines)
        po_number = self._extract_po_number(extracted_lines)
        tax_info = self._extract_taxes_and_totals(paddle_full_text, extracted_lines)

        # 4. GSTIN Extraction & Validation
        paddle_entities = DualEngineBenchmarker.extract_key_entities(paddle_full_text)
        tesseract_entities = DualEngineBenchmarker.extract_key_entities(tesseract_full_text)

        gstins = paddle_entities.get("gstin", [])
        if parties.get("vendor_gstin"):
            vendor_gstin = parties.get("vendor_gstin")
        elif parties.get("buyer_gstin"):
            vendor_gstin = None
        else:
            vendor_gstin = gstins[0] if len(gstins) > 0 else None

        if parties.get("buyer_gstin"):
            buyer_gstin = parties.get("buyer_gstin")
        elif vendor_gstin and len(gstins) > 1:
            buyer_gstin = gstins[1]
        else:
            buyer_gstin = gstins[0] if len(gstins) == 1 and not vendor_gstin else None

        if vendor_gstin and buyer_gstin and vendor_gstin == buyer_gstin:
            if parties.get("buyer_gstin") == buyer_gstin and parties.get("buyer_name"):
                vendor_gstin = None
            else:
                buyer_gstin = None

        # Cross-engine buyer resolution if missing from primary engine
        buyer_n = parties.get("buyer_name")
        if not buyer_n:
            m_tess = re.search(r"(?:Billed\s*to|Buyer|Customer|Consignee|Party\s*Name)[:\s]+([A-Za-z0-9\s.,&'-]{3,})", tesseract_full_text, re.IGNORECASE)
            if m_tess:
                cand = m_tess.group(1).strip()
                if len(cand) >= 3 and cand.lower() not in LABEL_BLACKLIST:
                    parties["buyer_name"] = cand

        vendor_gstin_audit = self.verifier.validate_gstin(vendor_gstin)
        buyer_gstin_audit = self.verifier.validate_gstin(buyer_gstin)

        # 5. Table Extraction
        table_res = self.table_extractor.extract_table(extracted_lines)
        line_items = table_res.get("line_items", [])
        extracted_subtotal = table_res.get("calculated_subtotal")

        # 6. Arithmetic Verification
        math_audit = self.verifier.verify_arithmetic(
            line_items=line_items,
            extracted_subtotal=extracted_subtotal,
            cgst=tax_info.get("cgst"),
            sgst=tax_info.get("sgst"),
            igst=tax_info.get("igst"),
            roundoff=tax_info.get("roundoff"),
            extracted_grand_total=tax_info.get("grand_total")
        )

        # 7. Consensus Auditing
        consensus_audit = self.verifier.audit_consensus(paddle_entities, tesseract_entities)

        total_pipeline_duration = round(time.time() - start_pipeline, 2)

        # Internal pipeline working data (preserving numeric types for verification)
        internal_data = {
            "file_name": img_path.name,
            "all_lines": extracted_lines,
            "invoice_metadata": {
                "invoice_number": inv_number,
                "invoice_date": inv_date,
                "due_date": due_date,
                "PO_number": po_number,
                "currency_code": "INR",
                "vendor": {
                    "name": parties.get("vendor_name"),
                    "gstin": vendor_gstin,
                    "gstin_valid": vendor_gstin_audit.get("valid", False),
                    "pan": parties.get("pan"),
                    "bank_details": parties.get("bank_details")
                },
                "buyer": {
                    "name": parties.get("buyer_name"),
                    "gstin": buyer_gstin,
                    "gstin_valid": buyer_gstin_audit.get("valid", False)
                }
            },
            "line_items": line_items,
            "financial_summary": {
                "subtotal": extracted_subtotal,
                "cgst": tax_info.get("cgst"),
                "sgst": tax_info.get("sgst"),
                "igst": tax_info.get("igst"),
                "roundoff": tax_info.get("roundoff"),
                "grand_total": math_audit.get("reconciled_grand_total") or tax_info.get("grand_total"),
                "grand_total_source": math_audit.get("grand_total_source", "ocr_extracted")
            },
            "verification": {
                "arithmetic_status": math_audit.get("overall_status"),
                "line_items_verified": math_audit.get("all_items_verified"),
                "subtotal_verified": math_audit.get("subtotal_verified"),
                "grand_total_verified": math_audit.get("grand_total_verified"),
                "dual_engine_consensus": consensus_audit.get("consensus_level"),
                "consensus_score_pct": consensus_audit.get("consensus_score"),
                "agreed_gstin": consensus_audit.get("agreed_gstin"),
                "agreed_total_amount": consensus_audit.get("agreed_total_amount")
            },
            "pipeline_performance": {
                "total_runtime_seconds": total_pipeline_duration,
                "paddle_gpu_runtime": paddle_res.get("runtime_seconds", 0.0),
                "tesseract_cpu_runtime": tesseract_res.get("runtime_seconds", 0.0),
                "paddle_average_confidence": paddle_res.get("average_confidence", 0.0)
            }
        }

        # Format into official output schema deliverable
        official_output = format_official_output(internal_data)

        # Diagnostic sidecar data (separate from official deliverable)
        audit_data = {
            "file_name": img_path.name,
            "verification": internal_data["verification"],
            "financial_summary": internal_data["financial_summary"],
            "pipeline_performance": internal_data["pipeline_performance"],
            "vendor_gstin_validation": vendor_gstin_audit,
            "buyer_gstin_validation": buyer_gstin_audit,
            "raw_metadata": internal_data["invoice_metadata"]
        }

        if return_diagnostics:
            return official_output, audit_data
        return official_output

    def process_folder(
        self,
        input_folder: Path,
        output_folder: Path,
        max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """Processes images in input_folder and saves structured JSON results."""
        in_p = Path(input_folder)
        out_p = Path(output_folder)
        out_p.mkdir(parents=True, exist_ok=True)

        images = sorted(list(in_p.glob("*.png")))
        if max_files:
            images = images[:max_files]

        processed_results = []
        for img in images:
            print(f"Processing invoice in pipeline: {img.name}...")
            official_res, audit_res = self.process_invoice(img, return_diagnostics=True)
            processed_results.append(official_res)

            # Save individual official JSON per invoice (strictly matching official schema)
            out_file = out_p / f"{img.stem}_invoice.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(official_res, f, indent=2)

            # Save individual sidecar audit JSON
            audit_file = out_p / f"{img.stem}_audit.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_res, f, indent=2)

        summary_file = out_p / "pipeline_summary.json"
        summary_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_processed": len(processed_results),
            "results": processed_results
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        return summary_data
