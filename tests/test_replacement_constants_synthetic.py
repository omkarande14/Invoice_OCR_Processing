# tests/test_replacement_constants_synthetic.py
"""
Synthetic layout tests demonstrating that replacement constants:
- 0.50 (mid_x page width ratio)
- 0.35 (header_y_limit page height ratio)
- 0.20 (missing description gap threshold)
- 0.35 (missing quantity+description gap threshold)
operate invariant to absolute image size, resolution, padding, and aspect ratio.
"""

import pytest
from app.ocr.self_healing import SelfHealingEngine
from app.ocr.table_extractor import TableExtractor

class TestReplacementConstantsSynthetic:
    """Verifies that ratio-based heuristics do not depend on fixed pixel dimensions."""

    @pytest.mark.parametrize("width,height,scale_factor", [
        (1000.0, 1400.0, 1.0),    # Standard 1000x1400
        (2500.0, 3500.0, 2.5),    # High-resolution 300 DPI scan
        (800.0, 2400.0, 0.8),     # Narrow aspect voucher
        (4000.0, 2500.0, 4.0),    # Landscape wide banner
    ])
    def test_self_healing_gstin_scale_invariance(self, width, height, scale_factor):
        """Seller on left (x < 0.5*W) and Buyer on right (x > 0.5*W) or lower (y > 0.35*H)."""
        engine = SelfHealingEngine()
        
        # Synthetic OCR tokens placed proportionally across the page
        seller_gstin = "27AAPFU0939F1ZV"
        buyer_gstin = "27AAPFU0939F1ZW"

        seller_box = [[width * 0.1, height * 0.1], [width * 0.3, height * 0.1], [width * 0.3, height * 0.15], [width * 0.1, height * 0.15]]
        buyer_box = [[width * 0.6, height * 0.1], [width * 0.85, height * 0.1], [width * 0.85, height * 0.15], [width * 0.6, height * 0.15]]
        footer_box = [[width * 0.1, height * 0.95], [width * 0.9, height * 0.95], [width * 0.9, height * 0.98], [width * 0.1, height * 0.98]]

        # Case A: Seller top-left, Buyer top-right
        ocr_lines_side_by_side = [
            {"text": f"GSTIN: {seller_gstin}", "bbox": seller_box},
            {"text": f"Customer GST: {buyer_gstin}", "bbox": buyer_box},
            {"text": "Authorised Signatory Total", "bbox": footer_box}
        ]

        meta = {"vendor": {}, "buyer": {}}
        res_side = engine.heal(meta, {}, [], all_ocr_lines=ocr_lines_side_by_side)
        healed_side = res_side["meta"]

        assert healed_side["vendor"].get("gstin") == seller_gstin, f"Failed at {width}x{height}: Seller GSTIN not assigned"
        assert healed_side["buyer"].get("gstin") == buyer_gstin, f"Failed at {width}x{height}: Buyer GSTIN not assigned"

        # Case B: Seller top-left (y < 0.35 * height), Buyer bottom-left (y > 0.35 * height)
        buyer_box_stacked = [[width * 0.1, height * 0.40], [width * 0.35, height * 0.40], [width * 0.35, height * 0.45], [width * 0.1, height * 0.45]]
        ocr_lines_stacked = [
            {"text": f"GSTIN: {seller_gstin}", "bbox": seller_box},
            {"text": f"Billed To: {buyer_gstin}", "bbox": buyer_box_stacked},
            {"text": "Authorised Signatory Total", "bbox": footer_box}
        ]

        meta_stacked = {"vendor": {}, "buyer": {}}
        res_stacked = engine.heal(meta_stacked, {}, [], all_ocr_lines=ocr_lines_stacked)
        healed_stacked = res_stacked["meta"]

        assert healed_stacked["vendor"].get("gstin") == seller_gstin, f"Stacked layout failed at {width}x{height}"
        assert healed_stacked["buyer"].get("gstin") == buyer_gstin, f"Stacked layout failed at {width}x{height}"

    @pytest.mark.parametrize("table_width,table_left", [
        (600.0, 50.0),
        (1200.0, 50.0),
        (2400.0, 100.0),
        (3600.0, 150.0)
    ])
    def test_table_extractor_gap_ratio_synthetic(self, table_width, table_left):
        """Verifies column inference thresholds: narrow (<0.20), medium (0.20-0.35), wide (>0.35)."""
        extractor = TableExtractor()

        def make_items(gap_width):
            b_min = table_left + gap_width
            b_max = b_min + 80.0
            a_min = table_left + table_width - 80.0
            a_max = table_left + table_width
            return [
                {
                    "text": "BATCH",
                    "box": [[b_min, 300.0], [b_max, 300.0], [b_max, 320.0], [b_min, 320.0]]
                },
                {
                    "text": "AMOUNT",
                    "box": [[a_min, 300.0], [a_max, 300.0], [a_max, 320.0], [a_min, 320.0]]
                }
            ]

        # 1. Narrow gap (< 0.20 of table width): No missing columns inferred
        narrow_gap = table_width * 0.10
        hdr_narrow = extractor._find_table_header(make_items(narrow_gap))
        assert hdr_narrow is not None
        names_narrow = {c["name"] for c in hdr_narrow["columns"]}
        assert "description" not in names_narrow, "Narrow gap should not infer description"
        assert "quantity" not in names_narrow, "Narrow gap should not infer quantity"

        # 2. Medium gap (0.20 < gap <= 0.35 of table width): Description inferred, quantity omitted
        med_gap = table_width * 0.25
        hdr_med = extractor._find_table_header(make_items(med_gap))
        assert hdr_med is not None
        names_med = {c["name"] for c in hdr_med["columns"]}
        assert "description" in names_med, "Medium gap must infer description"
        assert "quantity" not in names_med, "Medium gap should not infer quantity"

        # 3. Wide gap (> 0.35 of table width): Both quantity and description inferred
        wide_gap = table_width * 0.45
        hdr_wide = extractor._find_table_header(make_items(wide_gap))
        assert hdr_wide is not None
        names_wide = {c["name"] for c in hdr_wide["columns"]}
        assert "description" in names_wide, "Wide gap must infer description"
        assert "quantity" in names_wide, "Wide gap must infer quantity"

        # Check ordering: quantity must precede description, which must precede batch
        ordered_names = [c["name"] for c in hdr_wide["columns"]]
        assert ordered_names.index("quantity") < ordered_names.index("description") < ordered_names.index("batch")
