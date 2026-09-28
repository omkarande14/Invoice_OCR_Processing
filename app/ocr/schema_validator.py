# app/ocr/schema_validator.py

from typing import Dict, Any, List, Optional

HEADER_KEYS_SPEC: List[str] = [
    "invoice_id",
    "invoice_number",
    "invoice_date",
    "due_date",
    "customer_name",
    "customer_gstin",
    "seller_name",
    "seller_gstin",
    "currency_code",
    "PO_number",
    "DC_date",
    "DC_number",
    "invoice_amount",
    "round_off",
    "total_gst_rate",
    "total_quantity",
    "total_cgst_rate",
    "total_cgst_amount",
    "total_sgst_rate",
    "total_sgst_amount",
    "total_igst_rate",
    "total_igst_amount",
    "total_gst_amount",
    "items"
]

ITEM_KEYS_SPEC: List[str] = [
    "description",
    "Pack",
    "Batch",
    "quantity",
    "free_quantity",
    "total_quantity",
    "free_item_yn",
    "unit_price",
    "total_price",
    "reference_number",
    "hsn_sac",
    "item_code",
    "expiry_date",
    "Discount",
    "Discount_type",
    "Value",
    "Gst%",
    "MRP",
    "cgst_rate",
    "cgst_amount",
    "sgst_rate",
    "sgst_amount",
    "igst_rate",
    "igst_amount",
    "GST_AMT",
    "taxable_value"
]

class SchemaValidator:
    """
    Automated validator to enforce 100% structural conformance against the Team Lead's exact reference template:
    - 23 header keys in exact sequence + 'items' as the 24th key
    - 26 item keys in exact sequence for every line item
    - Disallows missing, renamed, misordered, or unexpected keys
    """

    def __init__(self, header_spec: Optional[List[str]] = None, item_spec: Optional[List[str]] = None):
        self.header_spec = header_spec or HEADER_KEYS_SPEC
        self.item_spec = item_spec or ITEM_KEYS_SPEC

    def validate(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []

        if not isinstance(output_json, dict):
            return {
                "valid": False,
                "status": "FAIL",
                "errors": ["Top-level JSON is not a dictionary"],
                "header_conformance": {"missing": [], "extra": [], "misordered": []},
                "items_conformance": []
            }

        actual_header_keys = list(output_json.keys())

        # 1. Header Keys Conformance
        missing_header = [k for k in self.header_spec if k not in actual_header_keys]
        extra_header = [k for k in actual_header_keys if k not in self.header_spec]
        
        # Check order
        expected_filtered = [k for k in self.header_spec if k in actual_header_keys]
        actual_filtered = [k for k in actual_header_keys if k in self.header_spec]
        misordered_header = []
        if expected_filtered != actual_filtered:
            for idx, (exp, act) in enumerate(zip(expected_filtered, actual_filtered)):
                if exp != act:
                    misordered_header.append(f"Position {idx}: expected '{exp}', got '{act}'")

        if missing_header:
            errors.append(f"Missing header keys: {missing_header}")
        if extra_header:
            errors.append(f"Unexpected extra header keys: {extra_header}")
        if misordered_header:
            errors.append(f"Misordered header keys: {misordered_header}")

        # 2. Items Array Check
        items = output_json.get("items")
        items_conformance = []
        if items is None or not isinstance(items, list):
            errors.append("'items' field is missing or not a list")
        else:
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"Item #{idx + 1} is not a dictionary")
                    continue
                actual_item_keys = list(item.keys())
                missing_item = [k for k in self.item_spec if k not in actual_item_keys]
                extra_item = [k for k in actual_item_keys if k not in self.item_spec]
                
                exp_item_filtered = [k for k in self.item_spec if k in actual_item_keys]
                act_item_filtered = [k for k in actual_item_keys if k in self.item_spec]
                misordered_item = []
                if exp_item_filtered != act_item_filtered:
                    for p_idx, (exp, act) in enumerate(zip(exp_item_filtered, act_item_filtered)):
                        if exp != act:
                            misordered_item.append(f"Position {p_idx}: expected '{exp}', got '{act}'")

                if missing_item or extra_item or misordered_item:
                    items_conformance.append({
                        "item_index": idx + 1,
                        "missing": missing_item,
                        "extra": extra_item,
                        "misordered": misordered_item
                    })
                    if missing_item:
                        errors.append(f"Item #{idx + 1} missing keys: {missing_item}")
                    if extra_item:
                        errors.append(f"Item #{idx + 1} unexpected extra keys: {extra_item}")
                    if misordered_item:
                        errors.append(f"Item #{idx + 1} misordered keys: {misordered_item}")

        is_valid = len(errors) == 0
        return {
            "valid": is_valid,
            "status": "PASS" if is_valid else "FAIL",
            "errors": errors,
            "header_conformance": {
                "missing": missing_header,
                "extra": extra_header,
                "misordered": misordered_header
            },
            "items_count": len(items) if isinstance(items, list) else 0,
            "items_conformance": items_conformance
        }
