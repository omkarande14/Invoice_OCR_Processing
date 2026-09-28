# app/ocr/semantic_ontology.py
"""
Centralized Semantic Synonym Ontology and Token Normalization.
Provides alias dictionaries, fuzzy matching, and term parsers
to eliminate fragile regex patterns across diverse invoice layouts and ERPs.
"""

import re
from typing import List, Optional, Set, Dict

# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------

def normalize_text_token(text: str) -> str:
    """Strips punctuation, collapses whitespace, and lowercases text."""
    if not text:
        return ""
    # Replace non-alphanumeric chars (except whitespace) with space
    cleaned = re.sub(r"[^a-zA-Z0-9\s]+", " ", text.lower())
    return " ".join(cleaned.split())

# ---------------------------------------------------------------------------
# Semantic Aliases
# ---------------------------------------------------------------------------

INVOICE_NUMBER_ALIASES: List[str] = [
    "invoice no", "inv no", "invoice number", "inv number",
    "bill no", "bill number", "tax invoice no", "tax invoice number",
    "invoice #", "inv #", "bill #", "cash memo", "cash memo no",
    "tax invoice", "retail invoice", "sale invoice"
]

INVOICE_DATE_ALIASES: List[str] = [
    "invoice date", "inv date", "bill date", "date of invoice",
    "invoice dt", "inv dt", "bill dt", "dated", "date:"
]

DUE_DATE_ALIASES: List[str] = [
    "due date", "due dt", "pay by", "payment due", "valid till",
    "due on", "payment terms", "credit days", "due date:", "due dt:"
]

PO_NUMBER_ALIASES: List[str] = [
    "po no", "po number", "order no", "order number", "p o no",
    "purchase order no", "purchase order", "customer po", "ord no", "po #"
]

DC_NUMBER_ALIASES: List[str] = [
    "dc no", "dc number", "delivery challan no", "challan no",
    "ch no", "d c no", "d.c. no", "delivery note"
]

BUYER_ANCHORS: List[str] = [
    "details of receiver (billed to)", "details of receiver", "billed to",
    "bill to", "buyer", "receiver", "customer", "party name", "customer name",
    "consignee", "buyer details", "purchaser", "client", "m/s", "messrs"
]

SELLER_ANCHORS: List[str] = [
    "sold by", "supplier", "vendor", "billed from", "consignor",
    "dispatch from", "seller", "from:"
]

PAYABLE_TOTAL_ALIASES: List[str] = [
    "to pay", "topay", "net payable", "net amount payable", "amount payable",
    "grand total", "total payable", "total amount", "invoice amount",
    "bill amount", "total bill", "final amount", "balance payable",
    "round off net", "net bill amount", "round off amount"
]

SUBTOTAL_ALIASES: List[str] = [
    "sub total", "subtotal", "taxable value", "taxable amount",
    "total before tax", "gross amount", "gross total", "item total"
]

ROUNDOFF_ALIASES: List[str] = [
    "round off", "roundoff", "r.off", "roff", "rnd off", "rounding"
]

# Blacklist of keywords that should NEVER be accepted as customer/vendor names
GLOBAL_PARTY_NAME_BLACKLIST: Set[str] = {
    "batch", "expiry", "rate", "amount", "qty", "quantity", "mrp", "hsn",
    "code", "pcode", "item", "description", "tax", "cgst", "sgst", "igst",
    "total", "subtotal", "invoice", "original", "duplicate", "triplicate",
    "customer copy", "supplier copy", "transporter copy", "details", "receiver",
    "billed", "to", "from", "terms", "conditions", "page", "sr no", "sl no",
    "for orders", "for orders and account", "orders and account", "statement"
}

# ---------------------------------------------------------------------------
# Table Column Semantic Mappings
# ---------------------------------------------------------------------------

COLUMN_ONTOLOGY: Dict[str, List[str]] = {
    "sr_no": ["sr no", "sr.no", "sl no", "sl.no", "s.no", "s no", "#"],
    "item_code": [
        "pcode", "p.code", "item code", "product code", "product id",
        "item id", "item no", "product no", "mat code", "material code",
        "art no", "article no"
    ],
    "description": [
        "description", "product description", "particulars", "item name",
        "product name", "item description", "items", "products", "medicine"
    ],
    "pack": ["pack", "packaging", "pkg", "unit", "uom"],
    "hsn_code": ["hsn", "hsn/sac", "hsn code", "sac code", "hsn / sac", "tariff"],
    "batch": ["batch", "batch no", "b.no", "b. no", "lot", "lot no"],
    "expiry": ["exp", "expiry", "exp date", "exp dt", "val", "validity"],
    "quantity": ["qty", "quantity", "billed qty", "qnty", "pcs", "nos"],
    "free_quantity": ["free", "free qty", "free quantity", "scheme", "sch", "f.qty", "fqty"],
    "rate": ["rate", "unit price", "price", "unit rate", "cost", "billing rate"],
    "mrp": ["mrp", "m.r.p", "m.r.p.", "maximum retail price", "max retail price"],
    "discount": ["disc", "disc%", "discount", "disc %", "discount %", "trd disc", "sch%"],
    "amount": ["amount", "total", "value", "net amount", "total amount", "taxable amt", "taxable value"],
    "gst_pct": ["gst%", "gst %", "tax%", "tax %", "gst rate", "cgst%", "sgst%", "igst%"]
}

# ---------------------------------------------------------------------------
# Matcher Functions
# ---------------------------------------------------------------------------

def matches_any_alias(text: str, aliases: List[str]) -> bool:
    """Checks if normalized text contains or starts with any of the aliases."""
    norm = normalize_text_token(text)
    if not norm:
        return False
    for alias in aliases:
        norm_alias = normalize_text_token(alias)
        if norm == norm_alias or norm.startswith(norm_alias) or f" {norm_alias} " in f" {norm} ":
            return True
    return False

def parse_credit_terms_days(text: str) -> Optional[int]:
    """
    Parses payment term duration in days from text.
    e.g. 'Credit: 7 Days' -> 7, 'Terms: 30 days net' -> 30, 'within 15 days' -> 15.
    """
    if not text:
        return None
    patterns = [
        r"(?:credit\s*(?:days|period)?|terms|due\s*in|within)[:\s]*(\d{1,3})\s*(?:days|day)?",
        r"(\d{1,3})\s*(?:days|day)\s*(?:credit|terms|net)",
        r"(?:net)\s*(\d{1,3})"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                days = int(m.group(1))
                if 0 < days <= 365:
                    return days
            except ValueError:
                pass
    return None
