# app/ocr/output_formatter.py

import calendar
import re
from typing import Dict, Any, List, Optional, Union
from app.ocr.self_healing import SelfHealingEngine

def format_decimal(val: Any) -> Optional[str]:
    """Formats numeric value to 2-decimal string (e.g. 5439.0 -> '5439.00')."""
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ""))
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return None

def format_number(val: Any) -> Optional[Union[int, float]]:
    """Formats numeric rate/quantity as integer if whole, else float."""
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ""))
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return None

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

def normalize_date_slashes(date_str: Optional[str]) -> Optional[str]:
    """Converts any date format (DD-MM-YYYY, DD/Aug/26, 27-Aug-2026, YYYY-MM-DD) to standard DD/MM/YYYY."""
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = date_str.strip()
    
    # 1. Numeric: DD/MM/YYYY or DD-MM-YY or DD.MM.YYYY
    m_num = re.match(r"^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{2,4})$", cleaned)
    if m_num:
        d, mo, yr = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))
        if yr < 100:
            yr += 2000
        return f"{d:02d}/{mo:02d}/{yr:04d}"

    # 2. Alpha month: 27-Aug-26, 27/Aug/2026, 27 Aug 2026
    m_alpha = re.match(r"^(\d{1,2})[-\/.\s]([A-Za-z]{3,9})[-\/.\s](\d{2,4})$", cleaned)
    if m_alpha:
        d = int(m_alpha.group(1))
        m_name = m_alpha.group(2).lower()
        yr = int(m_alpha.group(3))
        if yr < 100:
            yr += 2000
        if m_name in MONTH_MAP:
            mo = MONTH_MAP[m_name]
            return f"{d:02d}/{mo:02d}/{yr:04d}"

    # 3. ISO format: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.match(r"^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})$", cleaned)
    if m_iso:
        yr, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        return f"{d:02d}/{mo:02d}/{yr:04d}"

    # 4. Embedded date pattern in text (e.g. '27-Aug-26' inside longer string)
    m_embed = re.search(r"\b(\d{1,2})[-\/.]([A-Za-z]{3,9}|\d{1,2})[-\/.](\d{2,4})\b", cleaned)
    if m_embed:
        d = int(m_embed.group(1))
        m_part = m_embed.group(2)
        yr = int(m_embed.group(3))
        if yr < 100:
            yr += 2000
        if m_part.isdigit():
            mo = int(m_part)
            return f"{d:02d}/{mo:02d}/{yr:04d}"
        elif m_part.lower() in MONTH_MAP:
            mo = MONTH_MAP[m_part.lower()]
            return f"{d:02d}/{mo:02d}/{yr:04d}"

    return cleaned.replace("-", "/")

def expand_expiry_date(exp_str: Optional[str]) -> Optional[str]:
    """
    Expands MM/YY to DD/MM/YYYY using the last calendar day of that month.
    e.g. '2/29' -> '28/02/2029', '11/27' -> '30/11/2027'.
    """
    if not exp_str or not isinstance(exp_str, str):
        return None
    exp_clean = exp_str.strip()
    # Check MM/YY or M/YY or MM/YYYY
    m = re.match(r"^(\d{1,2})[\/\-](\d{2,4})$", exp_clean)
    if m:
        month = int(m.group(1))
        year_part = m.group(2)
        year = 2000 + int(year_part) if len(year_part) == 2 else int(year_part)
        if 1 <= month <= 12:
            last_day = calendar.monthrange(year, month)[1]
            return f"{last_day:02d}/{month:02d}/{year:04d}"

    # If full date already
    m_full = re.match(r"^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})$", exp_clean)
    if m_full:
        d, mo, yr = int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3))
        if yr < 100:
            yr += 2000
        return f"{d:02d}/{mo:02d}/{yr:04d}"

    return exp_clean

def format_official_output(internal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms internal pipeline data into the Team Lead's exact official schema.
    Enforces exact key names, key sequence, types (string vs number vs null),
    and flat structure with 'items' as the final key.
    """
    meta = internal_data.get("invoice_metadata", {})
    fin = internal_data.get("financial_summary", {})
    raw_items = internal_data.get("line_items", [])
    all_lines = internal_data.get("all_lines") or internal_data.get("extracted_lines", [])

    # Active Mathematical & Ledger Self-Healing
    healed = SelfHealingEngine.heal(meta, fin, raw_items, all_lines)
    meta = healed["meta"]
    vendor = meta.get("vendor", {})
    buyer = meta.get("buyer", {})
    fin = healed["fin"]
    raw_items = healed["items"]

    # Format line items first so header totals/rates can be computed
    formatted_items = []
    total_qty_sum = 0
    item_gst_rates = []
    item_cgst_rates = []
    item_sgst_rates = []

    for item in raw_items:
        qty_num = format_number(item.get("quantity"))
        free_qty_num = format_number(item.get("free_quantity", 0))
        if free_qty_num is None:
            free_qty_num = 0
        free_yn = "1" if free_qty_num > 0 else "0"

        # If promotional/free item (rate == 0 or amount == 0) and qty_num was identical to free_qty_num:
        if (item.get("rate") == 0.0 or item.get("amount") == 0.0) and qty_num == free_qty_num and free_qty_num > 0:
            qty_num = 0

        item_total_qty = (qty_num or 0) + (free_qty_num or 0)
        total_qty_sum += item_total_qty

        cgst_r = format_number(item.get("cgst_rate"))
        sgst_r = format_number(item.get("sgst_rate"))
        igst_r = format_number(item.get("igst_rate"))

        cgst_amt_val = item.get("cgst_amount")
        sgst_amt_val = item.get("sgst_amount")
        igst_amt_val = item.get("igst_amount")

        # Sanitize corrupted rates (> 40%) into amounts
        if cgst_r is not None and cgst_r > 40.0:
            if cgst_amt_val is None:
                cgst_amt_val = cgst_r
            cgst_r = None
        if sgst_r is not None and sgst_r > 40.0:
            if sgst_amt_val is None:
                sgst_amt_val = sgst_r
            sgst_r = None

        # If item has direct gst_pct and no split cgst/sgst rates:
        raw_gst_pct = item.get("gst_pct")
        if raw_gst_pct is not None and float(raw_gst_pct) <= 40.0 and cgst_r is None and sgst_r is None and igst_r is None:
            half = float(raw_gst_pct) / 2.0
            cgst_r = format_number(half)
            sgst_r = format_number(half)
            item_gst_pct = format_number(raw_gst_pct)
        elif cgst_r is not None and sgst_r is not None:
            item_gst_pct = format_number(cgst_r + sgst_r)
        elif igst_r is not None and igst_r <= 40.0:
            item_gst_pct = igst_r
        else:
            item_gst_pct = None

        if cgst_r is not None and cgst_r <= 40.0:
            item_cgst_rates.append(cgst_r)
        if sgst_r is not None and sgst_r <= 40.0:
            item_sgst_rates.append(sgst_r)
        if item_gst_pct is not None and item_gst_pct <= 40.0:
            item_gst_rates.append(item_gst_pct)

        # Item total price fallback (amount -> taxable_value -> qty * rate)
        item_total_price = item.get("amount")
        if item_total_price is None:
            item_total_price = item.get("taxable_value")
        if item_total_price is None and qty_num is not None and item.get("rate") is not None:
            try:
                item_total_price = round(float(qty_num) * float(item["rate"]), 2)
            except (ValueError, TypeError):
                pass

        # Determine taxable base for GST calculation
        taxable_base = item.get("taxable_value")
        if taxable_base is None:
            taxable_base = item_total_price
        taxable_base_float = None
        if taxable_base is not None:
            try:
                taxable_base_float = float(str(taxable_base).replace(",", ""))
            except (ValueError, TypeError):
                pass

        # Derive / calculate CGST, SGST, IGST amounts if null
        if taxable_base_float is not None:
            if cgst_amt_val is None and cgst_r is not None and cgst_r <= 40.0:
                cgst_amt_val = round(taxable_base_float * (float(cgst_r) / 100.0), 2)
            if sgst_amt_val is None and sgst_r is not None and sgst_r <= 40.0:
                sgst_amt_val = round(taxable_base_float * (float(sgst_r) / 100.0), 2)
            if igst_amt_val is None and igst_r is not None and igst_r <= 40.0:
                igst_amt_val = round(taxable_base_float * (float(igst_r) / 100.0), 2)

        # Combined item GST amount
        item_gst_amt_float = None
        if cgst_amt_val is not None or sgst_amt_val is not None or igst_amt_val is not None:
            item_gst_amt_float = (float(cgst_amt_val) if cgst_amt_val is not None else 0.0) + \
                                 (float(sgst_amt_val) if sgst_amt_val is not None else 0.0) + \
                                 (float(igst_amt_val) if igst_amt_val is not None else 0.0)
        elif item_gst_pct is not None and taxable_base_float is not None:
            item_gst_amt_float = round(taxable_base_float * (float(item_gst_pct) / 100.0), 2)

        # Taxable value formatted
        taxable_val_formatted = format_decimal(item.get("taxable_value"))
        if taxable_val_formatted is None and taxable_base_float is not None:
            taxable_val_formatted = format_decimal(taxable_base_float)

        # Construct each item dictionary with keys in exact sequence
        formatted_item = {
            "description": item.get("description"),
            "Pack": item.get("unit"),
            "Batch": item.get("batch"),
            "quantity": qty_num,
            "free_quantity": free_qty_num,
            "total_quantity": item_total_qty,
            "free_item_yn": free_yn,
            "unit_price": format_decimal(item.get("rate")),
            "total_price": format_decimal(item_total_price),
            "reference_number": item.get("reference_number", None),
            "hsn_sac": item.get("hsn_code"),
            "item_code": item.get("item_code"),
            "expiry_date": expand_expiry_date(item.get("expiry_date")),
            "Discount": format_decimal(item.get("discount_pct")),
            "Discount_type": item.get("discount_type", None),
            "Value": format_decimal(item.get("value")) if item.get("value") is not None else None,
            "Gst%": item_gst_pct,
            "MRP": format_decimal(item.get("mrp")),
            "cgst_rate": cgst_r,
            "cgst_amount": format_decimal(cgst_amt_val),
            "sgst_rate": sgst_r,
            "sgst_amount": format_decimal(sgst_amt_val),
            "igst_rate": igst_r,
            "igst_amount": format_decimal(igst_amt_val),
            "GST_AMT": format_decimal(item_gst_amt_float),
            "taxable_value": taxable_val_formatted
        }
        formatted_items.append(formatted_item)

    # Compute header-level uniform rates (approved Option A: null if mixed)
    def uniform_or_null(rate_list: List[Union[int, float]]) -> Optional[Union[int, float]]:
        if not rate_list:
            return None
        return rate_list[0] if len(set(rate_list)) == 1 else None

    header_gst_rate = uniform_or_null(item_gst_rates)
    header_cgst_rate = uniform_or_null(item_cgst_rates)
    header_sgst_rate = uniform_or_null(item_sgst_rates)

    # Header tax amounts: fallback to sum of line items if footer is missing / zero
    line_cgst_sum = sum(float(i["cgst_amount"]) for i in formatted_items if i.get("cgst_amount"))
    line_sgst_sum = sum(float(i["sgst_amount"]) for i in formatted_items if i.get("sgst_amount"))
    line_igst_sum = sum(float(i["igst_amount"]) for i in formatted_items if i.get("igst_amount"))

    cgst_tot = fin.get("cgst")
    if (cgst_tot is None or float(cgst_tot) == 0.0) and line_cgst_sum > 0:
        cgst_tot = line_cgst_sum

    sgst_tot = fin.get("sgst")
    if (sgst_tot is None or float(sgst_tot) == 0.0) and line_sgst_sum > 0:
        sgst_tot = line_sgst_sum

    igst_tot = fin.get("igst")
    if (igst_tot is None or float(igst_tot) == 0.0) and line_igst_sum > 0:
        igst_tot = line_igst_sum

    total_gst_amount = None
    if cgst_tot is not None or sgst_tot is not None or igst_tot is not None:
        total_gst_amount = format_decimal((cgst_tot or 0.0) + (sgst_tot or 0.0) + (igst_tot or 0.0))
    elif formatted_items:
        line_gst_sum = sum(float(i["GST_AMT"]) for i in formatted_items if i.get("GST_AMT"))
        if line_gst_sum > 0:
            total_gst_amount = format_decimal(line_gst_sum)

    # Format invoice & due dates
    inv_date = normalize_date_slashes(meta.get("invoice_date"))
    due_date = normalize_date_slashes(meta.get("due_date"))

    # Build official output dictionary with exact top-level sequence
    official_output = {
        "invoice_id": meta.get("invoice_number"),
        "invoice_number": meta.get("invoice_number"),
        "invoice_date": inv_date,
        "due_date": due_date,
        "customer_name": buyer.get("name"),
        "customer_gstin": buyer.get("gstin"),
        "seller_name": vendor.get("name"),
        "seller_gstin": vendor.get("gstin"),
        "currency_code": meta.get("currency_code", "INR"),
        "PO_number": meta.get("PO_number"),
        "DC_date": normalize_date_slashes(meta.get("DC_date")),
        "DC_number": meta.get("DC_number", None),
        "invoice_amount": format_decimal(fin.get("grand_total")),
        "round_off": format_decimal(fin.get("roundoff")),
        "total_gst_rate": header_gst_rate,
        "total_quantity": total_qty_sum,
        "total_cgst_rate": header_cgst_rate,
        "total_cgst_amount": format_decimal(cgst_tot) or "0.00",
        "total_sgst_rate": header_sgst_rate,
        "total_sgst_amount": format_decimal(sgst_tot) or "0.00",
        "total_igst_rate": format_number(fin.get("igst_rate")) if fin.get("igst_rate") is not None else None,
        "total_igst_amount": format_decimal(igst_tot) if igst_tot is not None else "0.00",
        "total_gst_amount": total_gst_amount or "0.00",
        "items": formatted_items
    }

    return official_output
