# app/rrsi/normalize.py
"""
Type-Aware Normalization and Matching Utilities for Invoice OCR Fields.
1. Invoice & PO Numbers: Case and whitespace normalization only (never numeric coercion).
2. Dates: ISO-8601 parsing and locale-swap detection (DD/MM vs MM/DD).
3. Amounts: Decimal representation with configurable absolute and relative tolerances.
4. Vendor/Customer Names: Casefold, punctuation strip, legal suffix normalization, exact + fuzzy token-set matching.
5. GSTIN: Exact matching and base-36 checksum validation according to the official GST specification.
"""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Tuple, Set


# =============================================================================
# 1. Invoice & PO Numbers
# =============================================================================
def normalize_invoice_number(val: Any) -> str:
    """
    Normalizes invoice/PO numbers by stripping leading/trailing whitespace
    and standardizing internal whitespace and uppercase.
    CRITICAL: NEVER converts to integer or float ("04531" != "4531").
    """
    if val is None:
        return ""
    s = str(val).strip()
    # Normalize multiple whitespace to single space
    s = " ".join(s.split())
    return s.upper()


def match_invoice_number(exp: Any, act: Any) -> Tuple[bool, Optional[str]]:
    """Compares expected and actual invoice numbers without numeric coercion."""
    n_exp = normalize_invoice_number(exp)
    n_act = normalize_invoice_number(act)

    if not n_exp and not n_act:
        return True, None
    if n_exp == n_act:
        return True, None
    return False, "invoice_number_mismatch"


# =============================================================================
# 2. Dates (ISO-8601 & Locale Swaps)
# =============================================================================
DATE_PATTERNS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d-%B-%Y",
    "%m/%d/%Y",
    "%m-%d-%Y"
]


def normalize_date(val: Any, default_format: Optional[str] = None) -> Optional[str]:
    """
    Parses various date formats into standard ISO-8601 string (YYYY-MM-DD).
    Returns None if unparseable.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    # Already ISO-8601
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    if default_format:
        try:
            return datetime.strptime(s, default_format).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try standard patterns
    for pat in DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, pat)
            # Normalize 2-digit years to 2000s
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def match_date(
    exp: Any,
    act: Any,
    default_format: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Compares two dates in ISO-8601.
    If mismatch occurs, tests if swapping Day and Month produces the expected date.
    Returns:
        (is_match, error_subtype)
        error_subtype can be None, "date_locale_swap", or "date_mismatch".
    """
    iso_exp = normalize_date(exp)
    iso_act = normalize_date(act, default_format=default_format)

    if not iso_exp and not iso_act:
        return True, None
    if not iso_exp or not iso_act:
        return False, "date_mismatch"

    if iso_exp == iso_act:
        return True, None

    # Check for DD/MM vs MM/DD locale swap
    y_exp, m_exp, d_exp = [int(x) for x in iso_exp.split("-")]
    y_act, m_act, d_act = [int(x) for x in iso_act.split("-")]

    if y_exp == y_act and m_exp == d_act and d_exp == m_act:
        return False, "date_locale_swap"

    return False, "date_mismatch"


# =============================================================================
# 3. Currency Amounts
# =============================================================================
def normalize_amount(val: Any) -> Optional[Decimal]:
    """
    Converts amount strings or numbers to Decimal, stripping currency symbols,
    commas, and whitespace.
    """
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    s = str(val).strip()
    if not s:
        return None

    # Remove currency symbols (INR, Rs, $, etc.) and commas
    cleaned = re.sub(r"[₹\$\€\£]|(?i:inr|rs\.?)\s*", "", s).replace(",", "").strip()
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def match_amount(
    exp: Any,
    act: Any,
    rel_tol: Decimal = Decimal("0.001"),
    abs_tol: Decimal = Decimal("1.00")
) -> Tuple[bool, Optional[str]]:
    """
    Compares amounts within configurable relative and absolute tolerance.
    Tolerance condition: |act - exp| <= max(abs_tol, rel_tol * |exp|)
    """
    d_exp = normalize_amount(exp)
    d_act = normalize_amount(act)

    if d_exp is None and d_act is None:
        return True, None
    if d_exp is None or d_act is None:
        return False, "amount_mismatch"

    diff = abs(d_act - d_exp)
    allowed_tolerance = max(abs_tol, rel_tol * abs(d_exp))

    if diff <= allowed_tolerance:
        return True, None
    return False, "amount_mismatch"


# =============================================================================
# 4. Vendor & Party Names
# =============================================================================
LEGAL_SUFFIX_MAP = {
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd.": "pvt ltd",
    "pvt. ltd": "pvt ltd",
    "private": "pvt",
    "pvt.": "pvt",
    "limited": "ltd",
    "ltd.": "ltd",
    "corporation": "corp",
    "corp.": "corp",
    "incorporated": "inc",
    "inc.": "inc",
    "llp.": "llp"
}


def normalize_vendor_name(val: Any) -> str:
    """
    Normalizes company/vendor names:
    - Casefold (lowercase)
    - Replaces punctuation with space
    - Standardizes legal suffixes (Pvt Ltd, Corp, Inc, LLP)
    - Collapses whitespace
    """
    if val is None:
        return ""
    s = str(val).lower().strip()

    # Replace punctuation with spaces
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())

    # Map full legal suffix phrases first
    for long_suffix, short_suffix in LEGAL_SUFFIX_MAP.items():
        pattern = r"\b" + re.escape(long_suffix) + r"\b"
        s = re.sub(pattern, short_suffix, s)

    return " ".join(s.split())


def _token_set_ratio(s1: str, s2: str) -> float:
    """Computes Dice similarity over token sets."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    intersection = len(t1 & t2)
    return (2.0 * intersection) / (len(t1) + len(t2))


def match_vendor_name(
    exp: Any,
    act: Any,
    fuzzy_threshold: float = 0.85
) -> Tuple[bool, bool, float]:
    """
    Returns:
        (exact_match, fuzzy_match, similarity_score)
    """
    n1 = normalize_vendor_name(exp)
    n2 = normalize_vendor_name(act)

    if not n1 and not n2:
        return True, True, 1.0
    if not n1 or not n2:
        return False, False, 0.0

    if n1 == n2:
        return True, True, 1.0

    # If concatenated words match exactly (e.g. health care vs healthcare)
    compact_1 = n1.replace(" ", "")
    compact_2 = n2.replace(" ", "")
    if compact_1 == compact_2:
        return False, True, 0.98

    # Token set ratio: common tokens vs remaining tokens
    t1 = set(n1.split())
    t2 = set(n2.split())
    intersection = t1 & t2
    diff1 = t1 - intersection
    diff2 = t2 - intersection

    # Calculate token dice score
    token_dice = (2.0 * len(intersection)) / (len(t1) + len(t2))

    # Also compute sequence matcher similarity on sorted tokens
    sorted_s1 = " ".join(sorted(t1))
    sorted_s2 = " ".join(sorted(t2))
    from difflib import SequenceMatcher
    char_sim = SequenceMatcher(None, sorted_s1, sorted_s2).ratio()

    score = max(token_dice, char_sim)
    fuzzy_match = score >= fuzzy_threshold
    return False, fuzzy_match, round(score, 3)


# =============================================================================
# 5. GSTIN Checksum Validation
# =============================================================================
GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")


def validate_gstin_checksum(gstin: str) -> bool:
    """
    Validates Indian GSTIN 15-character format and check digit using the
    official Base-36 Luhn-variant checksum algorithm.
    """
    if not gstin or not isinstance(gstin, str):
        return False

    clean_gstin = gstin.strip().upper()
    if not GSTIN_REGEX.match(clean_gstin):
        return False

    # Multipliers for characters 0 through 13 (alternating 1, 2, ...)
    multipliers = [1, 2] * 7

    total_sum = 0
    for i in range(14):
        char = clean_gstin[i]
        code_point = GSTIN_CHARS.index(char)
        product = code_point * multipliers[i]
        # Sum quotient and remainder in base 36
        quotient = product // 36
        remainder = product % 36
        total_sum += quotient + remainder

    expected_remainder = total_sum % 36
    check_code_point = (36 - expected_remainder) % 36
    expected_check_char = GSTIN_CHARS[check_code_point]

    return clean_gstin[14] == expected_check_char


def match_gstin(exp: Any, act: Any) -> Tuple[bool, bool, Optional[str]]:
    """
    Validates and matches GSTIN.
    Returns:
        (is_match: bool, is_valid_checksum: bool, error_subtype: Optional[str])
    """
    s_exp = str(exp).strip().upper() if exp else ""
    s_act = str(act).strip().upper() if act else ""

    if not s_exp and not s_act:
        return True, True, None
    if not s_exp or not s_act:
        return False, False, "gstin_mismatch"

    is_valid = validate_gstin_checksum(s_act)
    if s_exp == s_act:
        return True, is_valid, None

    return False, is_valid, "gstin_mismatch"
