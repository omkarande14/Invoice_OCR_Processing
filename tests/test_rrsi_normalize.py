# tests/test_rrsi_normalize.py
import pytest
from decimal import Decimal
from app.rrsi.normalize import (
    normalize_invoice_number,
    match_invoice_number,
    normalize_date,
    match_date,
    normalize_amount,
    match_amount,
    normalize_vendor_name,
    match_vendor_name,
    validate_gstin_checksum,
    match_gstin
)

# ---------------------------------------------------------------------------
# 1. Invoice Number Tests: Whitespace and case only, NEVER numeric coercion
# ---------------------------------------------------------------------------
def test_invoice_number_never_coerces_numerically():
    # "04531" != "4531"
    assert normalize_invoice_number("04531") == "04531"
    assert normalize_invoice_number("4531") == "4531"
    matched, err = match_invoice_number("04531", "4531")
    assert matched is False
    assert err == "invoice_number_mismatch"

def test_invoice_number_case_and_whitespace():
    assert normalize_invoice_number("  wmp/B/04531  ") == "WMP/B/04531"
    matched, err = match_invoice_number("WMP/B/04531", "  wmp/b/04531  ")
    assert matched is True
    assert err is None

# ---------------------------------------------------------------------------
# 2. Date Tests: ISO-8601 parsing and date_locale_swap classification
# ---------------------------------------------------------------------------
def test_date_iso_matching():
    # Standard format: 20/08/2026 -> 2026-08-20
    assert normalize_date("20/08/2026") == "2026-08-20"
    assert normalize_date("20-08-2026") == "2026-08-20"
    assert normalize_date("2026-08-20") == "2026-08-20"

    matched, err = match_date("2026-08-20", "20/08/2026")
    assert matched is True
    assert err is None

def test_date_locale_swap_detected():
    # Ground truth is 05 July 2026 ("2026-07-05").
    # Prediction parsed as May 07 2026 ("2026-05-07") because of MM/DD vs DD/MM swap.
    matched, err = match_date("2026-07-05", "05/07/2026", default_format="%m/%d/%Y")
    # If the prediction assumed MM/DD/YYYY, "05/07/2026" yielded May 7, which differs from July 5
    matched, err = match_date("2026-07-05", "2026-05-07")
    assert matched is False
    assert err == "date_locale_swap"

def test_date_genuine_mismatch():
    matched, err = match_date("2026-07-25", "2026-07-10")
    assert matched is False
    assert err == "date_mismatch"

# ---------------------------------------------------------------------------
# 3. Amount Tests: Decimal with configurable relative + absolute tolerance
# ---------------------------------------------------------------------------
def test_amount_decimal_parsing_and_matching():
    # Currency symbols and commas
    amt = normalize_amount("Rs. 1,23,456.78")
    assert amt == Decimal("123456.78")

    # Within absolute tolerance (default abs_tol=1.0)
    matched, err = match_amount("100.50", "101.00", abs_tol=Decimal("1.00"), rel_tol=Decimal("0.001"))
    assert matched is True

    # Exceeding tolerance
    matched, err = match_amount("100.00", "102.50", abs_tol=Decimal("1.00"), rel_tol=Decimal("0.001"))
    assert matched is False
    assert err == "amount_mismatch"

# ---------------------------------------------------------------------------
# 4. Name Tests: Casefold, punctuation strip, legal suffix normalization, exact vs fuzzy
# ---------------------------------------------------------------------------
def test_vendor_name_normalization_and_fuzzy():
    # Normalizing "Acme Health Care Private Limited" vs "acme health care pvt. ltd."
    n1 = normalize_vendor_name("Acme Health Care Private Limited")
    n2 = normalize_vendor_name("acme health care pvt. ltd.")
    assert n1 == "acme health care pvt ltd"
    assert n2 == "acme health care pvt ltd"

    exact, fuzzy, score = match_vendor_name("Acme Health Care Private Limited", "acme health care pvt. ltd.")
    assert exact is True
    assert fuzzy is True
    assert score == 1.0

    # Slight variation: exact is False, but fuzzy is True
    exact2, fuzzy2, score2 = match_vendor_name("Acme Health Care Pvt Ltd", "Acme Healthcare Pvt Ltd")
    assert exact2 is False
    assert fuzzy2 is True
    assert score2 >= 0.85

# ---------------------------------------------------------------------------
# 5. GSTIN Tests: Exact match and checksum validity
# ---------------------------------------------------------------------------
def test_gstin_checksum_validity():
    # Valid Indian GSTIN: 27AAPFU0939F1ZV
    assert validate_gstin_checksum("27AAPFU0939F1ZV") is True
    # Invalid checksum char (altered last char to 'W')
    assert validate_gstin_checksum("27AAPFU0939F1ZW") is False

def test_gstin_matching():
    matched, valid_chk, err = match_gstin("27AAPFU0939F1ZV", "27AAPFU0939F1ZV")
    assert matched is True
    assert valid_chk is True
    assert err is None

    matched, valid_chk, err = match_gstin("27AAPFU0939F1ZV", "27AAPFU0939F1ZW")
    assert matched is False
    assert err == "gstin_mismatch"
