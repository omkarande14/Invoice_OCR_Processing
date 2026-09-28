import re
from typing import List, Dict, Any, Optional, Tuple

class TableExtractor:
    """
    Extracts structured line-item tables from OCR bounding boxes using
    spatial geometric column alignment and row clustering.
    Reconciles row arithmetic: Quantity * Rate == Amount.
    """

    # Keyword mappings to canonical column identifiers
    HEADER_KEYWORDS = {
        "sr": "sr_no",
        "sl": "sr_no",
        "s.no": "sr_no",
        "s.n": "sr_no",
        "item": "description",
        "product": "description",
        "description": "description",
        "particulars": "description",
        "hsn": "hsn_code",
        "sac": "hsn_code",
        "qty": "quantity",
        "quantity": "quantity",
        "rate": "rate",
        "price": "rate",
        "mrp": "mrp",
        "amount": "amount",
        "net": "amount",
        "value": "taxable_value",
        "taxable": "taxable_value",
        "batch": "batch",
        "exp": "expiry",
        "disc": "discount",
        "dis": "discount",
        "discount": "discount",
        "free": "free_quantity",
        "f.qty": "free_quantity",
        "fr.qty": "free_quantity",
        "sch.qty": "free_quantity",
        "sch": "free_quantity",
        "scheme": "free_quantity",
        "per": "unit",
        "pack": "unit",
        "unit": "unit",
        "cgst": "cgst",
        "sgst": "sgst",
        "igst": "igst",
        "gst": "gst_pct",
        "rack": "item_code",
        "code": "item_code",
        "pcode": "item_code",
        "p.code": "item_code"
    }

    # Keywords marking the boundary where line items end and summaries/taxes begin
    TABLE_STOP_KEYWORDS = [
        "sub total", "subtotal", "gross amt", "gross", "net amt", "net amount", "total",
        "out put", "output", "output cgst", "output sgst", "cgst amt", "sgst amt",
        "amount chargeable", "amount in words", "rs.in word", "bank details",
        "declaration", "authorised signatory", "authorized signatory",
        "terms of delivery", "gst rate", "taxable value", "msg :", "short supply",
        "order no", "order no.", "ord date", "sch amt", "disc amt",
        "osamt", "osamt:", "custgihe",
        "cn ret", "cnre", "sman:", "sman", "po/phrmcy", "po/no", "competent person",
        "checked by", "taxable cgst", "amountdisc"
    ]

    def __init__(self, y_row_tolerance: float = 16.0):
        self.y_row_tolerance = y_row_tolerance

    @staticmethod
    def _get_box_coords(box: List[List[float]]) -> Dict[str, float]:
        """Calculates bounding box dimensions and centroid."""
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "x_center": float((xmin + xmax) / 2.0),
            "y_center": float((ymin + ymax) / 2.0),
            "height": float(ymax - ymin),
            "width": float(xmax - xmin)
        }

    def _find_table_header(self, elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identifies the table header row and establishes column X coordinates."""
        header_candidates = []
        for el in elements:
            text_lower = el["text"].lower().strip()

            # Skip product codes like SR-02-0391 from table header consideration
            if re.search(r"\b[A-Za-z]{2,4}-\d{2}-\d{4}\b", el["text"]):
                continue

            # Check for merged header tokens like 'RATEDIS' (Rate + Discount)
            if "rate" in text_lower and ("dis" in text_lower or "disc" in text_lower):
                coords = self._get_box_coords(el["box"])
                w_half = coords["width"] / 2.0
                header_candidates.append({
                    "col_type": "rate",
                    "text": "rate",
                    "coords": {
                        **coords,
                        "xmax": coords["xmin"] + w_half,
                        "x_center": coords["xmin"] + (w_half / 2.0),
                        "width": w_half
                    }
                })
                header_candidates.append({
                    "col_type": "discount",
                    "text": "discount",
                    "coords": {
                        **coords,
                        "xmin": coords["xmin"] + w_half,
                        "x_center": coords["xmin"] + w_half + (w_half / 2.0),
                        "width": w_half
                    }
                })
                continue

            # Check for merged header tokens like 'Disc%Gst%' (Discount + GST%)
            if ("disc" in text_lower or "dis" in text_lower) and "gst" in text_lower:
                coords = self._get_box_coords(el["box"])
                w_half = coords["width"] / 2.0
                header_candidates.append({
                    "col_type": "discount",
                    "text": "discount",
                    "coords": {
                        **coords,
                        "xmax": coords["xmin"] + w_half,
                        "x_center": coords["xmin"] + (w_half / 2.0),
                        "width": w_half
                    }
                })
                header_candidates.append({
                    "col_type": "gst_pct",
                    "text": "gst_pct",
                    "coords": {
                        **coords,
                        "xmin": coords["xmin"] + w_half,
                        "x_center": coords["xmin"] + w_half + (w_half / 2.0),
                        "width": w_half
                    }
                })
                continue

            norm_text = text_lower.replace(".", "")
            words = re.findall(r"[A-Za-z]+", text_lower)
            words_norm = re.findall(r"[A-Za-z]+", norm_text)
            all_words = list(dict.fromkeys(words + words_norm + [norm_text]))
            for w in all_words:
                # If 'code' is part of 'HSN Code', do not map to item_code
                if w == "code" and any("hsn" in e["text"].lower() for e in elements if abs(min(p[1] for p in e["box"]) - min(p[1] for p in el["box"])) < 35 and abs(min(p[0] for p in e["box"]) - min(p[0] for p in el["box"])) < 50):
                    continue
                # If 'gst' is part of '(GST)' under or near 'HSN Code', do not map to gst_pct
                if w == "gst" and any("hsn" in e["text"].lower() for e in elements if abs(min(p[1] for p in e["box"]) - min(p[1] for p in el["box"])) < 40 and abs(min(p[0] for p in e["box"]) - min(p[0] for p in el["box"])) < 60):
                    continue
                if w in self.HEADER_KEYWORDS:
                    col_type = self.HEADER_KEYWORDS[w]
                    coords = self._get_box_coords(el["box"])
                    header_candidates.append({
                        "col_type": col_type,
                        "text": el["text"],
                        "coords": coords
                    })
                    break

        if not header_candidates:
            return None

        # Group header items by vertical alignment (within 45px for wide/tilted headers)
        header_bands = []
        for cand in header_candidates:
            matched_band = False
            for band in header_bands:
                if abs(band["y_center"] - cand["coords"]["y_center"]) <= 45.0:
                    band["items"].append(cand)
                    band["cols"].add(cand["col_type"])
                    band["y_center"] = sum(i["coords"]["y_center"] for i in band["items"]) / len(band["items"])
                    matched_band = True
                    break
            if not matched_band:
                header_bands.append({
                    "items": [cand],
                    "cols": {cand["col_type"]},
                    "y_center": cand["coords"]["y_center"]
                })

        # Find band with the most distinct column types
        header_bands.sort(key=lambda b: len(b["cols"]), reverse=True)
        best_band = header_bands[0] if header_bands else None

        if not best_band or len(best_band["cols"]) < 2:
            return None

        y_header_top = min(i["coords"]["ymin"] for i in best_band["items"])
        header_ymaxes = sorted(i["coords"]["ymax"] for i in best_band["items"])
        median_ymax = header_ymaxes[len(header_ymaxes) // 2]
        y_header_bottom = min(max(header_ymaxes), median_ymax + 20.0)

        # Create structured column anchors sorted by X center
        # Deduplicate multiple items for same col_type by taking average position
        columns = {}
        for item in best_band["items"]:
            c_type = item["col_type"]
            if c_type not in columns:
                columns[c_type] = {
                    "name": c_type,
                    "x_center": item["coords"]["x_center"],
                    "xmin": item["coords"]["xmin"],
                    "xmax": item["coords"]["xmax"]
                }
            else:
                # Merge column bounds
                columns[c_type]["xmin"] = min(columns[c_type]["xmin"], item["coords"]["xmin"])
                columns[c_type]["xmax"] = max(columns[c_type]["xmax"], item["coords"]["xmax"])
                columns[c_type]["x_center"] = (columns[c_type]["xmin"] + columns[c_type]["xmax"]) / 2.0

        # If hsn_code and gst_pct columns overlap closely in X, drop gst_pct in favor of hsn_code
        if "hsn_code" in columns and "gst_pct" in columns:
            if abs(columns["hsn_code"]["x_center"] - columns["gst_pct"]["x_center"]) < 45:
                del columns["gst_pct"]

        col_list = sorted(list(columns.values()), key=lambda c: c["x_center"])

        # Infer missing description and quantity columns to the left of the first column if wide gap exists
        col_names = {c["name"] for c in col_list}
        first_col_xmin = col_list[0]["xmin"] if col_list else 0.0
        table_right = max(c["xmax"] for c in col_list) if col_list else 1000.0
        table_left = 50.0
        gap_width = first_col_xmin - table_left

        if gap_width > 0.20 * table_right:
            inferred = []
            if "quantity" not in col_names and gap_width > 0.35 * table_right:
                split_x = table_left + (gap_width * 0.45)
                inferred.append({
                    "name": "quantity",
                    "x_center": (table_left + split_x) / 2.0,
                    "xmin": table_left,
                    "xmax": split_x
                })
                inferred.append({
                    "name": "description",
                    "x_center": (split_x + first_col_xmin) / 2.0,
                    "xmin": split_x,
                    "xmax": first_col_xmin
                })
            elif "description" not in col_names:
                inferred.append({
                    "name": "description",
                    "x_center": (table_left + first_col_xmin) / 2.0,
                    "xmin": table_left,
                    "xmax": first_col_xmin
                })
            col_list = sorted(inferred + col_list, key=lambda c: c["x_center"])

        return {
            "y_top": y_header_top,
            "y_bottom": y_header_bottom,
            "columns": col_list
        }

    def _find_table_footer_y(self, elements: List[Dict[str, Any]], y_header_bottom: float) -> float:
        """Locates the bottom boundary of the table by detecting summary/footer keywords."""
        stop_candidates = []
        for el in elements:
            coords = self._get_box_coords(el["box"])
            if coords["ymin"] > y_header_bottom:
                txt_lower = el["text"].lower().strip()
                for stop_kw in self.TABLE_STOP_KEYWORDS:
                    if stop_kw in txt_lower:
                        stop_candidates.append(coords["ymin"])
                        break

        if stop_candidates:
            return min(stop_candidates)
        all_ymax = [self._get_box_coords(el["box"])["ymax"] for el in elements]
        return max(all_ymax) if all_ymax else 9999.0

    @staticmethod
    def _parse_numeric(text: str) -> Optional[float]:
        """Extracts floating point values from strings."""
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        try:
            val = float(cleaned)
            return val
        except ValueError:
            return None

    def _assign_cell_to_column(self, cell_coords: Dict[str, float], columns: List[Dict[str, Any]], text: Optional[str] = None) -> str:
        """Finds the best matching column for a given cell using interval partitioning and semantic data-type validation."""
        x_c = cell_coords["x_center"]
        sorted_cols = sorted(columns, key=lambda c: c["x_center"])

        # Determine column boundaries via Voronoi intervals between centers
        assigned_col = sorted_cols[0]["name"]
        for i, col in enumerate(sorted_cols):
            left_bound = (sorted_cols[i-1]["x_center"] + col["x_center"]) / 2.0 if i > 0 else 0.0
            right_bound = (col["x_center"] + sorted_cols[i+1]["x_center"]) / 2.0 if i < len(sorted_cols) - 1 else float("inf")

            if left_bound <= x_c < right_bound:
                assigned_col = col["name"]
                break

        # Semantic data-type validation:
        # If assigned column is numeric (quantity, rate, amount, mrp, gst_pct) but text has NO digits and contains words,
        # reassign to description if a description column exists
        if text:
            numeric_cols = {"quantity", "rate", "amount", "mrp", "gst_pct", "free_quantity"}
            if assigned_col in numeric_cols:
                has_digits = bool(re.search(r"\d", text))
                has_alpha = bool(re.search(r"[A-Za-z]{3,}", text))
                if not has_digits and has_alpha:
                    for col in sorted_cols:
                        if col["name"] == "description":
                            return "description"

        return assigned_col

    def extract_table(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts structured rows and columns using spatial alignment.
        """
        if not elements:
            return {"status": "empty", "line_items": []}

        header_info = self._find_table_header(elements)
        if not header_info:
            return {"status": "no_header_detected", "line_items": []}

        y_header_bottom = header_info["y_bottom"]
        y_footer_top = self._find_table_footer_y(elements, y_header_bottom)
        columns = header_info["columns"]

        # Collect header box coordinates to exclude exact header elements from table body
        header_boxes = []
        for el in elements:
            coords = self._get_box_coords(el["box"])
            if coords["ymin"] >= header_info["y_top"] - 10 and coords["ymax"] <= header_info["y_bottom"] + 10:
                t_low = el["text"].lower().strip()
                norm_t = t_low.replace(".", "")
                words = re.findall(r"[A-Za-z]+", t_low) + re.findall(r"[A-Za-z]+", norm_t)
                if any(w in self.HEADER_KEYWORDS for w in words) and not re.search(r"\b\d{2,}\b", t_low):
                    header_boxes.append(coords)

        table_elements = []
        for el in elements:
            coords = self._get_box_coords(el["box"])
            if coords["ymin"] < header_info["y_top"] - 5:
                continue
            if coords["ymin"] >= y_footer_top:
                continue
            # Skip if this element is one of the recognized header column labels
            is_header = False
            for hb in header_boxes:
                if abs(coords["xmin"] - hb["xmin"]) < 10 and abs(coords["ymin"] - hb["ymin"]) < 10:
                    is_header = True
                    break
            if is_header:
                continue

            table_elements.append({
                "text": el["text"].strip(),
                "confidence": el.get("confidence", 1.0),
                "box": el["box"],
                "coords": coords
            })

        # Dynamically scale row clustering tolerance based on document font/line height
        if table_elements:
            avg_height = sum(e["coords"]["height"] for e in table_elements) / len(table_elements)
            effective_row_tolerance = max(10.0, min(36.0, avg_height * 0.75))
        else:
            effective_row_tolerance = self.y_row_tolerance

        # Sort top-to-bottom, left-to-right
        table_elements.sort(key=lambda e: (e["coords"]["y_center"], e["coords"]["xmin"]))

        # Cluster elements into horizontal rows with column collision protection
        SINGLE_VALUE_COLS = {
            "amount", "rate", "quantity", "free_quantity", "mrp", 
            "batch", "expiry", "hsn_code", "gst_pct", "discount", 
            "item_code", "sr_no"
        }

        raw_rows = []
        for el in table_elements:
            col = self._assign_cell_to_column(el["coords"], columns, el.get("text"))
            matched_row = False
            for row in raw_rows:
                if abs(row["y_center"] - el["coords"]["y_center"]) <= effective_row_tolerance:
                    row_cols = [self._assign_cell_to_column(e["coords"], columns, e.get("text")) for e in row["elements"]]
                    if col in SINGLE_VALUE_COLS and col in row_cols:
                        continue
                    row["elements"].append(el)
                    row["y_center"] = sum(e["coords"]["y_center"] for e in row["elements"]) / len(row["elements"])
                    matched_row = True
                    break
            if not matched_row:
                raw_rows.append({
                    "y_center": el["coords"]["y_center"],
                    "elements": [el]
                })

        # Multi-line item sub-row merging:
        # If a row has product info (description/code/qty) but no money (rate/amount),
        # and the immediately following row (within 70px) has money, merge them!
        merged_rows = []
        skip_next = False
        for i in range(len(raw_rows)):
            if skip_next:
                skip_next = False
                continue
            curr_r = raw_rows[i]
            curr_cols = {self._assign_cell_to_column(e['coords'], columns, e['text']): e for e in curr_r['elements']}
            curr_has_money = "rate" in curr_cols or "amount" in curr_cols

            if not curr_has_money and i + 1 < len(raw_rows):
                next_r = raw_rows[i+1]
                next_cols = {self._assign_cell_to_column(e['coords'], columns, e['text']): e for e in next_r['elements']}
                next_has_money = "rate" in next_cols or "amount" in next_cols
                if next_has_money and abs(next_r["y_center"] - curr_r["y_center"]) <= 70:
                    merged_rows.append({
                        "y_center": next_r["y_center"],
                        "elements": curr_r["elements"] + next_r["elements"]
                    })
                    skip_next = True
                    continue
            merged_rows.append(curr_r)

        # Process each row into a structured line item
        line_items = []
        current_item = None

        for row in merged_rows:
            row["elements"].sort(key=lambda e: e["coords"]["xmin"])
            row_dict = {}
            for el in row["elements"]:
                col_name = self._assign_cell_to_column(el["coords"], columns)
                if col_name not in row_dict:
                    row_dict[col_name] = el["text"]
                else:
                    row_dict[col_name] += " " + el["text"]

            raw_text = " | ".join(e["text"] for e in row["elements"])
            row_text_lower = " ".join(e["text"].lower() for e in row["elements"])

            # 1. Stop if this row contains table boundary/summary keywords
            if any(stop_kw in row_text_lower for stop_kw in self.TABLE_STOP_KEYWORDS):
                break

            # 2. Check if this row is an item header row or a sub-item continuation
            has_qty = "quantity" in row_dict and self._parse_numeric(row_dict["quantity"]) is not None
            has_rate = "rate" in row_dict and self._parse_numeric(row_dict["rate"]) is not None
            has_amount = "amount" in row_dict and self._parse_numeric(row_dict["amount"]) is not None
            has_sr = "sr_no" in row_dict and bool(re.match(r"^\d{1,3}$", row_dict["sr_no"].strip()))

            # Deduce quantity via row arithmetic: Qty * Rate == Amount if quantity column wasn't explicitly assigned
            if not has_qty and has_rate and has_amount:
                r_val = self._parse_numeric(row_dict["rate"])
                a_val = self._parse_numeric(row_dict["amount"])
                if r_val and a_val and r_val > 0:
                    for el in row["elements"]:
                        num_m = re.match(r"^(\d{1,5})$", el["text"].strip())
                        if num_m:
                            q_cand = float(num_m.group(1))
                            if q_cand > 0 and abs(q_cand * r_val - a_val) < 1.0:
                                row_dict["quantity"] = str(q_cand)
                                has_qty = True
                                break

            # 3. Detect standalone subtotal / total rows:
            # E.g. A row containing an amount with no item description, quantity, or rate sitting at the bottom of line items
            if has_amount and not has_qty and not has_rate and not has_sr:
                if line_items or current_item:
                    desc_elements = [
                        e for e in row["elements"]
                        if self._assign_cell_to_column(e["coords"], columns) == "description"
                    ]
                    has_real_desc = any(re.search(r"[A-Za-z]{2,}", e["text"]) for e in desc_elements)
                    if not has_real_desc:
                        # Standalone subtotal/total row at bottom of table
                        break

            is_new_item = (has_qty and (has_rate or has_amount)) or (has_sr and (has_qty or has_rate or has_amount))

            if is_new_item:
                if current_item:
                    line_items.append(current_item)

                sr_no = int(row_dict["sr_no"].strip()) if has_sr else (len(line_items) + 1)
                desc = row_dict.get("description", "").strip()
                item_code = row_dict.get("item_code", "").strip() or None
                hsn = row_dict.get("hsn_code", "").strip() or None

                # Clean watermark strings out of product description (e.g. 'TRIPLICATEFOR SUPPLIER', 'ORIGINAL FOR BUYER')
                desc = re.sub(
                    r"\s*\(?(?:ORIGINAL|DUPLICATE|TRIPLICATE)\s*(?:FOR)?\s*(?:SUPPLIER|BUYER|TRANSPORTER|CUSTOMER)?\)?",
                    "",
                    desc,
                    flags=re.IGNORECASE
                ).strip()

                # Clean drug license numbers, company codes, or stray parentheses
                desc = re.sub(r"\b\d{2}\s*\d{5,}[A-Za-z0-9\.\s\/-]*", "", desc).strip()
                desc = re.sub(r"\bDL\s*NO[\s\S]*", "", desc, flags=re.IGNORECASE).strip()
                desc = re.sub(r"\s*\([A-Za-z0-9\/\s.-]+\)", "", desc).strip()

                code_m = re.match(r"^([A-Z]{2,4}-\d{2}-\d{4})(.*)", desc)
                if code_m and not item_code:
                    item_code = code_m.group(1).strip()
                    desc = code_m.group(2).strip()

                # Check if item_code or hsn contains composite HSN + PCODE (e.g. '30049029AL-02-3524')
                raw_code_check = item_code or hsn
                if raw_code_check:
                    comp_m = re.match(r"^(\d{4,8})\s*([A-Za-z]{1,4}[A-Za-z0-9\/-]{2,15}|(?=.*[A-Za-z])[A-Za-z0-9\/-]{3,15})$", raw_code_check)
                    if comp_m:
                        hsn = comp_m.group(1)
                        item_code = comp_m.group(2)

                if not item_code:
                    raw_code = row_dict.get("item_code", "").strip()
                    if raw_code and len(raw_code) >= 2 and not raw_code.lower().startswith(("gst", "tax", "batch", "qty", "hsn", "total")):
                        item_code = raw_code

                if not desc:
                    # Look for non-numeric tokens
                    non_nums = [e["text"] for e in row["elements"] if not re.match(r"^[\d\.,\/-]+$", e["text"])]
                    desc = " ".join(non_nums) if non_nums else f"Item {sr_no}"

                # Parse quantities, rates, and amounts
                qty = self._parse_numeric(row_dict.get("quantity", "")) if has_qty else None
                rate = self._parse_numeric(row_dict.get("rate", "")) if has_rate else None
                amount = self._parse_numeric(row_dict.get("amount", "")) if has_amount else None
                if not hsn:
                    hsn = row_dict.get("hsn_code", "").strip() or None
                if hsn:
                    hsn = re.sub(r"^\d{1,2}\s+", "", hsn).strip()

                # Clean unit if attached to quantity (e.g. '9 no' or '9 jar')
                unit = row_dict.get("unit", "").strip() or None
                if not unit and "quantity" in row_dict:
                    u_match = re.search(r"[A-Za-z]+", row_dict["quantity"])
                    if u_match:
                        unit = u_match.group(0).lower()

                # Free quantity extraction logic
                free_qty = 0.0
                if "free_quantity" in row_dict:
                    parsed_free = self._parse_numeric(row_dict["free_quantity"])
                    if parsed_free and parsed_free > 0:
                        free_qty = parsed_free

                # Check for compound quantity expressions like '20 + 2', '20+ 2', '10+1', or '20+'
                raw_qty_str = row_dict.get("quantity", "")
                compound_m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)?", raw_qty_str)
                if compound_m and compound_m.group(1):
                    base_qty = float(compound_m.group(1))
                    free_part = float(compound_m.group(2)) if compound_m.group(2) else 1.0
                    qty = base_qty
                    free_qty = free_part
                elif free_qty == 0.0:
                    # Also check description for embedded compound quantity (e.g. '20+ 2FLUDAC20MG' or '20+ 2')
                    desc_compound_m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)(?=[A-Za-z\s]|$)", desc)
                    if desc_compound_m:
                        base_qty = float(desc_compound_m.group(1))
                        free_part = float(desc_compound_m.group(2))
                        if qty is None or abs(qty - base_qty) < 0.01:
                            qty = base_qty
                            free_qty = free_part
                            # Clean compound quantity out of description
                            desc = re.sub(r"\b\d+(?:\.\d+)?\s*\+\s*\d+(?:\.\d+)?\s*", "", desc).strip()

                # Also detect 0-priced line items or promotional free items
                if (rate == 0.0 or amount == 0.0 or re.search(r"\bFREE\b|\bSCHEME\b|\bBONUS\b", desc, re.I)):
                    if qty is not None and qty > 0 and free_qty == 0:
                        free_qty = qty
                        qty = 0.0  # Paid quantity is 0 for promotional items

                # Mathematical reconciliation: Qty * Rate == Amount
                math_verified = False
                if qty is not None and rate is not None and amount is not None:
                    calc = round(qty * rate, 2)
                    math_verified = abs(calc - amount) <= 1.0 or (
                        # Account for taxable value after discount
                        "discount" in row_dict and amount <= calc
                    )

                # Extract auxiliary line-item fields (batch, expiry, mrp, discount, taxes, manufacturer)
                exp_m = re.search(r"\b(\d{1,2}\/\d{2,4})\b", raw_text)
                expiry = exp_m.group(1) if exp_m else row_dict.get("expiry")

                tokens = [t.strip() for t in raw_text.split("|")]
                if not item_code:
                    for t in tokens:
                        if re.match(r"^[A-Z]{2,4}-\d{2}-\d{4}$", t) or re.match(r"^[A-Z0-9\/-]{5,15}$", t):
                            if t != hsn and t != expiry and not re.match(r"^\d+\.\d{2}$", t):
                                item_code = t
                                break

                batch = row_dict.get("batch")
                if not batch:
                    for t in tokens:
                        if (re.match(r"^[A-Z0-9]{5,10}$", t) or re.match(r"^[A-Z0-9]{1,4}-[A-Z0-9]{2,6}$", t)) and t != hsn and t != item_code and t != expiry:
                            batch = t
                            break

                mfr = None
                for t in tokens:
                    if t in ["INTAS", "CIPLA", "SUN", "LUPIN", "ZYDUS", "MANKIND", "ALKEM", "TORRENT", "GLENMARK"] or (
                        re.match(r"^[A-Z]{3,8}$", t) and t not in ["VIAL", "AMP", "TAB", "CAP", "BOX", "PCS", "NOS", "EXP", "MRP", "RATE", "DISC", "SGST", "CGST", "IGST", "AMOUNT"]
                    ):
                        mfr = t
                        break

                nums = [float(t) for t in tokens if re.match(r"^\d+\.\d{2}$", t)]
                mrp = self._parse_numeric(row_dict.get("mrp", ""))
                disc = self._parse_numeric(row_dict.get("discount", ""))
                sgst_r = self._parse_numeric(row_dict.get("sgst", ""))
                sgst_a = None
                cgst_r = self._parse_numeric(row_dict.get("cgst", ""))
                cgst_a = None
                gst_pct = self._parse_numeric(row_dict.get("gst_pct", ""))

                if mrp is None and len(nums) >= 2:
                    mrp = nums[0]
                if disc is None and len(nums) >= 3 and nums[2] < 100.0:
                    disc = nums[2]
                if len(nums) >= 7:
                    if sgst_r is None:
                        sgst_r = nums[3]
                    sgst_a = nums[4]
                    if cgst_r is None:
                        cgst_r = nums[5]
                    cgst_a = nums[6]

                if not hsn:
                    for t in tokens:
                        if re.match(r"^\d{6,8}$", t) and t != batch and t != (str(int(qty)) if qty else ""):
                            hsn = t
                            break

                if amount is None:
                    taxable_val = self._parse_numeric(row_dict.get("taxable_value", ""))
                    if taxable_val is not None:
                        amount = taxable_val
                    elif qty is not None and rate is not None:
                        amount = round(qty * rate, 2)

                # Validate rates: GST rates cannot exceed 40%
                if cgst_r is not None and cgst_r > 40.0:
                    cgst_a = cgst_r
                    cgst_r = None
                if sgst_r is not None and sgst_r > 40.0:
                    sgst_a = sgst_r
                    sgst_r = None
                if gst_pct is not None:
                    if gst_pct > 40.0:
                        if not hsn and re.match(r"^\d{6,8}$", str(int(gst_pct))):
                            hsn = str(int(gst_pct))
                        gst_pct = None
                    elif cgst_r is None and sgst_r is None:
                        cgst_r = round(gst_pct / 2.0, 2)
                        sgst_r = round(gst_pct / 2.0, 2)

                if gst_pct is None and cgst_r is None and sgst_r is None:
                    for t in tokens:
                        if re.match(r"^\d{1,2}(?:\.0+)?$", t):
                            val = float(t)
                            if val in [5.0, 12.0, 18.0, 28.0]:
                                gst_pct = val
                                cgst_r = round(val / 2.0, 2)
                                sgst_r = round(val / 2.0, 2)
                                break

                # Calculate tax amounts from rates if not explicitly extracted
                taxable_calc_base = amount if amount is not None else ((qty * rate) if (qty is not None and rate is not None) else None)
                if taxable_calc_base is not None and disc is not None and 0 < disc < 100:
                    taxable_calc_base = round(taxable_calc_base * (1.0 - disc / 100.0), 2)
                if taxable_calc_base is not None and taxable_calc_base > 0:
                    if cgst_a is None and cgst_r is not None and cgst_r <= 40.0:
                        cgst_a = round(taxable_calc_base * (cgst_r / 100.0), 2)
                    if sgst_a is None and sgst_r is not None and sgst_r <= 40.0:
                        sgst_a = round(taxable_calc_base * (sgst_r / 100.0), 2)

                if cgst_a is not None and sgst_a is None and cgst_r == sgst_r:
                    sgst_a = cgst_a
                elif sgst_a is not None and cgst_a is None and cgst_r == sgst_r:
                    cgst_a = sgst_a

                total_qty = (qty or 0.0) + (free_qty or 0.0)

                current_item = {
                    "sr_no": sr_no,
                    "item_code": item_code,
                    "description": desc,
                    "hsn_code": hsn,
                    "quantity": qty,
                    "unit": unit,
                    "batch": batch,
                    "manufacturer": mfr,
                    "expiry_date": expiry,
                    "mrp": mrp,
                    "rate": rate,
                    "discount_pct": disc,
                    "gst_pct": gst_pct,
                    "sgst_rate": sgst_r,
                    "sgst_amount": sgst_a,
                    "cgst_rate": cgst_r,
                    "cgst_amount": cgst_a,
                    "amount": amount,
                    "free_quantity": free_qty,
                    "total_quantity": total_qty,
                    "math_verified": math_verified,
                    "raw_row": raw_text
                }
            else:
                # Continuation / multi-line description or notes
                if current_item:
                    # Only append elements that belong to description column or non-monetary text
                    desc_pieces = []
                    for e in row["elements"]:
                        col = self._assign_cell_to_column(e["coords"], columns, e.get("text"))
                        t = e["text"].strip()
                        is_monetary = bool(re.match(r"^[\d,.]+$", t))
                        is_watermark = bool(re.search(r"\b(?:ORIGINAL|DUPLICATE|TRIPLICATE|SUPPLIER|TRANSPORTER|ACKNOWLEDGE)\b", t, re.I))
                        if is_watermark or any(stop_kw in t.lower() for stop_kw in self.TABLE_STOP_KEYWORDS):
                            continue
                        if col == "description" or (not is_monetary and col not in ("amount", "rate", "quantity")):
                            desc_pieces.append(t)
                    if desc_pieces:
                        extra_text = " ".join(desc_pieces)
                        current_item["description"] += f" ({extra_text})"

        if current_item:
            line_items.append(current_item)

        total_extracted_amount = round(sum(i["amount"] for i in line_items if i.get("amount")), 2)
        all_math_verified = all(i["math_verified"] for i in line_items if i.get("math_verified") is not None)

        return {
            "status": "success",
            "columns_detected": [c["name"] for c in columns],
            "total_items": len(line_items),
            "line_items": line_items,
            "calculated_subtotal": total_extracted_amount,
            "table_math_verified": all_math_verified and len(line_items) > 0
        }
