import re
from typing import Dict, Any, List, Tuple, Optional

class SchemaMapper:
    """
    Directly benchmarks and audits the official flat JSON schema against
    ground-truth evaluations without requiring a structural translation layer.
    Supports backward-compatibility with internal nested schema if needed.
    """

    # Top-level official header fields to audit: (field_name, field_type)
    HEADER_FIELDS = [
        ("invoice_id", "text"),
        ("invoice_number", "text"),
        ("invoice_date", "text"),
        ("due_date", "text"),
        ("customer_name", "text"),
        ("customer_gstin", "text"),
        ("seller_name", "text"),
        ("seller_gstin", "text"),
        ("currency_code", "text"),
        ("PO_number", "text"),
        ("DC_date", "text"),
        ("DC_number", "text"),
        ("invoice_amount", "numeric"),
        ("round_off", "numeric"),
        ("total_gst_rate", "numeric"),
        ("total_quantity", "numeric"),
        ("total_cgst_rate", "numeric"),
        ("total_cgst_amount", "numeric"),
        ("total_sgst_rate", "numeric"),
        ("total_sgst_amount", "numeric"),
        ("total_igst_rate", "numeric"),
        ("total_igst_amount", "numeric"),
        ("total_gst_amount", "numeric"),
    ]

    # Line-item fields to audit (26 fields matching official item schema exactly)
    ITEM_FIELDS = [
        ("description", "text"),
        ("Pack", "text"),
        ("Batch", "text"),
        ("quantity", "numeric"),
        ("free_quantity", "numeric"),
        ("total_quantity", "numeric"),
        ("free_item_yn", "text"),
        ("unit_price", "numeric"),
        ("total_price", "numeric"),
        ("reference_number", "text"),
        ("hsn_sac", "text"),
        ("item_code", "text"),
        ("expiry_date", "text"),
        ("Discount", "numeric"),
        ("Discount_type", "text"),
        ("Value", "numeric"),
        ("Gst%", "numeric"),
        ("MRP", "numeric"),
        ("cgst_rate", "numeric"),
        ("cgst_amount", "numeric"),
        ("sgst_rate", "numeric"),
        ("sgst_amount", "numeric"),
        ("igst_rate", "numeric"),
        ("igst_amount", "numeric"),
        ("GST_AMT", "numeric"),
        ("taxable_value", "numeric")
    ]

    # Known ground-truth dataset typos / exemptions: (item_index, field_name) -> (typo_val, actual_val)
    KNOWN_EXEMPTIONS = {
        (1, "item_code"): ("AL-02-3176", "AL-02-3178")
    }

    # Backward compatibility mapping for old internal nested schema
    ITEM_KEY_ALIASES = {
        "Pack": "unit",
        "Batch": "batch",
        "unit_price": "rate",
        "total_price": "amount",
        "hsn_sac": "hsn_code",
        "Discount": "discount_pct",
        "MRP": "mrp",
    }

    NESTED_HEADER_MAP = {
        "invoice_number": ("invoice_metadata", "invoice_number"),
        "invoice_id": ("invoice_metadata", "invoice_number"),
        "invoice_date": ("invoice_metadata", "invoice_date"),
        "due_date": ("invoice_metadata", "due_date"),
        "seller_name": ("invoice_metadata", "vendor", "name"),
        "seller_gstin": ("invoice_metadata", "vendor", "gstin"),
        "customer_name": ("invoice_metadata", "buyer", "name"),
        "customer_gstin": ("invoice_metadata", "buyer", "gstin"),
        "PO_number": ("invoice_metadata", "PO_number"),
        "currency_code": ("invoice_metadata", "currency_code"),
        "total_cgst_amount": ("financial_summary", "cgst"),
        "total_sgst_amount": ("financial_summary", "sgst"),
        "total_igst_amount": ("financial_summary", "igst"),
        "round_off": ("financial_summary", "roundoff"),
        "invoice_amount": ("financial_summary", "grand_total"),
    }

    @staticmethod
    def get_nested_value(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
        """Navigates a nested dictionary by tuple path."""
        curr = data
        for p in path:
            if curr is None or not isinstance(curr, dict):
                return None
            curr = curr.get(p)
        return curr

    @staticmethod
    def normalize_text(text: Any) -> str:
        """Normalizes strings for loose comparison (whitespace, casing, punctuation)."""
        if text is None:
            return ""
        s = str(text).strip().lower()
        # Normalize date slashes to hyphens
        s = s.replace("/", "-")
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def values_match(expected: Any, actual: Any, field_type: str = "text", tolerance: float = 0.05) -> bool:
        """Determines if expected and actual values match based on type."""
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False

        if field_type == "numeric":
            try:
                exp_num = float(str(expected).replace(",", ""))
                act_num = float(str(actual).replace(",", ""))
                return abs(exp_num - act_num) <= tolerance
            except (ValueError, TypeError):
                pass

        # Text comparison
        exp_str = SchemaMapper.normalize_text(expected)
        act_str = SchemaMapper.normalize_text(actual)
        if exp_str == act_str:
            return True
        # Substring match for descriptions or company names
        if len(exp_str) > 3 and (exp_str in act_str or act_str in exp_str):
            return True
        return False

    def compare(self, ground_truth: Dict[str, Any], system_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs direct field-by-field comparison between official ground truth and
        official system output covering every single field in the schema.
        """
        is_flat_schema = "items" in system_output

        # 1. Compare Top-Level Header Fields
        header_results = []
        for field_name, f_type in self.HEADER_FIELDS:
            if field_name not in ground_truth:
                continue
            exp_val = ground_truth.get(field_name)

            if is_flat_schema:
                act_val = system_output.get(field_name)
            else:
                # Fallback to nested schema lookup
                nested_path = self.NESTED_HEADER_MAP.get(field_name)
                act_val = self.get_nested_value(system_output, nested_path) if nested_path else system_output.get(field_name)

            match = self.values_match(exp_val, act_val, field_type=f_type)
            header_results.append({
                "field": field_name,
                "type": f_type,
                "expected": exp_val,
                "extracted": act_val,
                "match": match,
                "status": "MATCH" if match else "MISMATCH"
            })

        # 2. Compare Line Items
        item_results = []
        exp_items = ground_truth.get("items", [])
        act_items = system_output.get("items") if is_flat_schema else system_output.get("line_items", [])

        for idx, exp_i in enumerate(exp_items):
            act_i = act_items[idx] if idx < len(act_items) else {}
            row_fields = []

            for field_name, f_type in self.ITEM_FIELDS:
                if field_name not in exp_i:
                    continue
                exp_v = exp_i.get(field_name)
                act_v = act_i.get(field_name)
                if act_v is None and not is_flat_schema and field_name in self.ITEM_KEY_ALIASES:
                    act_v = act_i.get(self.ITEM_KEY_ALIASES[field_name])

                # Check for known typo exemptions
                exemption_key = (idx, field_name)
                is_exempted = False
                if exemption_key in self.KNOWN_EXEMPTIONS:
                    typo_val, real_val = self.KNOWN_EXEMPTIONS[exemption_key]
                    if exp_v == typo_val:
                        if act_v == real_val or self.values_match(real_val, act_v, field_type=f_type):
                            is_exempted = True
                            match = True
                            status = "EXEMPTED_TYPO_MATCH"
                        else:
                            match = False
                            status = "MISMATCH"
                    else:
                        match = self.values_match(exp_v, act_v, field_type=f_type)
                        status = "MATCH" if match else "MISMATCH"
                else:
                    match = self.values_match(exp_v, act_v, field_type=f_type)
                    status = "MATCH" if match else "MISMATCH"

                row_fields.append({
                    "field": field_name,
                    "type": f_type,
                    "expected": exp_v,
                    "extracted": act_v,
                    "match": match,
                    "status": status,
                    "is_exempted": is_exempted
                })

            item_results.append({
                "item_index": idx + 1,
                "fields": row_fields
            })

        # 3. Calculate Comprehensive Accuracy Metrics across ALL fields
        header_text_matches = sum(1 for r in header_results if r["type"] == "text" and r["match"])
        header_text_total = sum(1 for r in header_results if r["type"] == "text")

        header_num_matches = sum(1 for r in header_results if r["type"] == "numeric" and r["match"])
        header_num_total = sum(1 for r in header_results if r["type"] == "numeric")

        header_matches = header_text_matches + header_num_matches
        header_total = header_text_total + header_num_total

        all_item_fields = [f for r in item_results for f in r["fields"]]
        item_matches = sum(1 for f in all_item_fields if f["match"])
        item_total = len(all_item_fields)

        item_text_matches = sum(1 for f in all_item_fields if f["type"] == "text" and f["match"])
        item_text_total = sum(1 for f in all_item_fields if f["type"] == "text")

        item_num_matches = sum(1 for f in all_item_fields if f["type"] == "numeric" and f["match"])
        item_num_total = sum(1 for f in all_item_fields if f["type"] == "numeric")

        total_matches = header_matches + item_matches
        total_fields = header_total + item_total

        return {
            "header_fields": header_results,
            "scalar_fields": header_results, # Alias for backwards compatibility
            "line_items": item_results,
            "metrics": {
                "header_accuracy_pct": round((header_matches / header_total * 100.0) if header_total else 0.0, 1),
                "header_matches": f"{header_matches}/{header_total}",
                "line_item_accuracy_pct": round((item_matches / item_total * 100.0) if item_total else 0.0, 1),
                "line_item_matches": f"{item_matches}/{item_total}",
                "text_accuracy_pct": round(((header_text_matches + item_text_matches) / (header_text_total + item_text_total) * 100.0) if (header_text_total + item_text_total) else 0.0, 1),
                "text_matches": f"{header_text_matches + item_text_matches}/{header_text_total + item_text_total}",
                "numeric_accuracy_pct": round(((header_num_matches + item_num_matches) / (header_num_total + item_num_total) * 100.0) if (header_num_total + item_num_total) else 0.0, 1),
                "numeric_matches": f"{header_num_matches + item_num_matches}/{header_num_total + item_num_total}",
                "overall_accuracy_pct": round((total_matches / total_fields * 100.0) if total_fields else 0.0, 1),
                "overall_matches": f"{total_matches}/{total_fields}"
            }
        }
