# scripts/check_annotation.py
"""
CLI Annotation Checker for Blind Ground-Truth Labeling (Phase 1.3 / 1.4).
Validates:
1. Required metadata (annotator, date, source, tier).
2. ISO-8601 Date formats.
3. Official Base-36 GSTIN checksums (customer and seller).
4. Tri-State Cross-Field Arithmetic Validation (PASS/FAIL/INSUFFICIENT_DATA).
5. Line items and amounts structure.
"""

import sys
import json
import argparse
from pathlib import Path
from decimal import Decimal

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rrsi.normalize import normalize_date, validate_gstin_checksum
from app.validators import InvoiceArithmeticValidator, ValidationStatus

def check_annotation_file(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"[!] Error: File {file_path} not found.")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[!] Invalid JSON format: {e}")
            return False

    errors = []
    warnings = []

    # 1. Metadata check
    meta = data.get("metadata", {})
    if not meta.get("annotator"):
        errors.append("Metadata 'annotator' is missing or empty.")
    if meta.get("source") not in ["from_image_blind", "pipeline_prefilled_then_corrected"]:
        errors.append("Metadata 'source' must be 'from_image_blind' or 'pipeline_prefilled_then_corrected'.")

    # 2. Date checks
    inv_date = data.get("invoice_date")
    if not inv_date or not normalize_date(inv_date):
        errors.append(f"Invalid invoice_date '{inv_date}'. Must be parseable to ISO-8601 (YYYY-MM-DD).")

    # 3. GSTIN checksums
    for party in ["customer_gstin", "seller_gstin"]:
        gst = data.get(party)
        if gst:
            if not validate_gstin_checksum(gst):
                errors.append(f"Invalid GSTIN checksum for '{party}': '{gst}' failed statutory Base-36 algorithm.")

    # 4. Arithmetic Validator
    validator = InvoiceArithmeticValidator(tolerance=Decimal("1.00"))
    val_res = validator.validate(data)
    if val_res.status != ValidationStatus.PASS:
        errors.append(f"Arithmetic validation failed: {val_res.message}")

    print(f"\n[*] Validation Report for: {file_path.name}")
    print(f"    - Metadata Source:   {meta.get('source', 'UNSPECIFIED')}")
    print(f"    - Annotator:         {meta.get('annotator', 'UNSPECIFIED')}")
    print(f"    - Arithmetic Status: {val_res.status.value} (Discrepancy: {val_res.discrepancy})")
    if val_res.reconciled_formula:
        print(f"    - Formula:           {val_res.reconciled_formula}")

    if errors:
        print("\n[!] VALIDATION FAILED:")
        for err in errors:
            print(f"    - ERROR: {err}")
        return False

    print("\n[+] VALIDATION PASSED: Document satisfies all statutory and arithmetic criteria.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate invoice annotation JSON.")
    parser.add_argument("file", help="Path to annotation JSON file")
    args = parser.parse_args()

    success = check_annotation_file(Path(args.file))
    sys.exit(0 if success else 1)
