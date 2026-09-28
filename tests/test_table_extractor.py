import pytest
from pathlib import Path
from app.ocr.table_extractor import TableExtractor

def test_table_extractor_synthetic():
    # Build synthetic bounding box elements representing a clean table
    # Columns at:
    # Description (x: 100-300), HSN (x: 400-500), Qty (x: 550-650), Rate (x: 700-800), Amount (x: 850-950)
    header_elements = [
        {"text": "Description of Goods", "box": [[100, 100], [300, 100], [300, 130], [100, 130]]},
        {"text": "HSN Code", "box": [[400, 100], [500, 100], [500, 130], [400, 130]]},
        {"text": "Quantity", "box": [[550, 100], [650, 100], [650, 130], [550, 130]]},
        {"text": "Rate", "box": [[700, 100], [800, 100], [800, 130], [700, 130]]},
        {"text": "Amount", "box": [[850, 100], [950, 100], [950, 130], [850, 130]]},
    ]

    # Row 1: Widget A, HSN 84713010, Qty 5, Rate 200, Amount 1000
    row1_elements = [
        {"text": "Widget A", "box": [[100, 160], [300, 160], [300, 190], [100, 190]]},
        {"text": "84713010", "box": [[400, 160], [500, 160], [500, 190], [400, 190]]},
        {"text": "5 pcs", "box": [[550, 160], [650, 160], [650, 190], [550, 190]]},
        {"text": "200.00", "box": [[700, 160], [800, 160], [800, 190], [700, 190]]},
        {"text": "1000.00", "box": [[850, 160], [950, 160], [950, 190], [850, 190]]},
    ]

    # Footer row
    footer_elements = [
        {"text": "Total", "box": [[700, 250], [800, 250], [800, 280], [700, 280]]},
        {"text": "1000.00", "box": [[850, 250], [950, 250], [950, 280], [850, 280]]},
    ]

    elements = header_elements + row1_elements + footer_elements

    extractor = TableExtractor()
    result = extractor.extract_table(elements)

    assert result["status"] == "success"
    assert result["total_items"] == 1
    item = result["line_items"][0]
    assert "Widget A" in item["description"]
    assert item["hsn_code"] == "84713010"
    assert item["quantity"] == 5.0
    assert item["rate"] == 200.0
    assert item["amount"] == 1000.0
    assert item["math_verified"] is True
    assert result["table_math_verified"] is True
    assert result["calculated_subtotal"] == 1000.0

def test_empty_elements():
    extractor = TableExtractor()
    res = extractor.extract_table([])
    assert res["status"] == "empty"
    assert res["line_items"] == []

def test_no_header_elements():
    extractor = TableExtractor()
    res = extractor.extract_table([
        {"text": "Some random text", "box": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    ])
    assert res["status"] == "no_header_detected"
    assert res["line_items"] == []

def test_last_line_item_does_not_absorb_totals_row_bleedthrough():
    """
    Regression test: ensures that when the last line item has a multi-line description
    and sits directly above a standalone subtotal/totals row (with or without a 'Total' label),
    the totals value (e.g. 28,800.00) is NOT absorbed into the product description.
    """
    header_elements = [
        {"text": "Description of Goods", "box": [[100, 100], [300, 100], [300, 130], [100, 130]]},
        {"text": "HSN Code", "box": [[400, 100], [500, 100], [500, 130], [400, 130]]},
        {"text": "Quantity", "box": [[550, 100], [650, 100], [650, 130], [550, 130]]},
        {"text": "Rate", "box": [[700, 100], [800, 100], [800, 130], [700, 130]]},
        {"text": "Amount", "box": [[850, 100], [950, 100], [950, 130], [850, 130]]},
    ]

    # Item 1: Nuprep Gel
    row1 = [
        {"text": "1 Nuprep Gel", "box": [[100, 150], [300, 150], [300, 180], [100, 180]]},
        {"text": "30067000", "box": [[400, 150], [500, 150], [500, 180], [400, 180]]},
        {"text": "9 no", "box": [[550, 150], [650, 150], [650, 180], [550, 180]]},
        {"text": "1400.00", "box": [[700, 150], [800, 150], [800, 180], [700, 180]]},
        {"text": "12600.00", "box": [[850, 150], [950, 150], [950, 180], [850, 180]]},
    ]

    # Item 2: Paste -Ten 20 (multi-line description)
    row2_main = [
        {"text": "2 Paste -Ten 20", "box": [[100, 200], [300, 200], [300, 230], [100, 230]]},
        {"text": "30067000", "box": [[400, 200], [500, 200], [500, 230], [400, 230]]},
        {"text": "9 jar", "box": [[550, 200], [650, 200], [650, 230], [550, 230]]},
        {"text": "1800.00", "box": [[700, 200], [800, 200], [800, 230], [700, 230]]},
        {"text": "16200.00", "box": [[850, 200], [950, 200], [950, 230], [850, 230]]},
    ]
    # Continuation row for Item 2 in description column
    row2_continuation = [
        {"text": "1 Box= 3 Jar (Weight-228 GM)", "box": [[100, 240], [350, 240], [350, 270], [100, 270]]},
    ]

    # Standalone Subtotal row sitting directly beneath the line items (no label, only amount)
    subtotal_row = [
        {"text": "28800.00", "box": [[850, 290], [950, 290], [950, 320], [850, 320]]},
    ]

    # Tax row below subtotal
    tax_row = [
        {"text": "Out Put CGST", "box": [[550, 340], [750, 340], [750, 370], [550, 370]]},
        {"text": "720.00", "box": [[850, 340], [950, 340], [950, 370], [850, 370]]},
    ]

    elements = header_elements + row1 + row2_main + row2_continuation + subtotal_row + tax_row

    extractor = TableExtractor()
    result = extractor.extract_table(elements)

    assert result["status"] == "success"
    assert result["total_items"] == 2
    item2 = result["line_items"][1]

    # Ensure the description contains the legitimate product spec but NOT the subtotal amount
    assert "Paste -Ten 20" in item2["description"]
    assert "1 Box= 3 Jar" in item2["description"]
    assert "28800" not in item2["description"]
    assert "28,800" not in item2["description"]
    assert item2["amount"] == 16200.0
    assert result["calculated_subtotal"] == 28800.0

def test_free_quantity_and_tax_derivation_in_table():
    """
    Validates:
    1. Compound quantity '20+ 2' in quantity column extracts paid=20, free=2, total=22
    2. Embedded '20+ 2' in description extracts paid=20, free=2, total=22 and cleans description
    3. Row-level tax derivation: calculates cgst_amount & sgst_amount from rates
    4. Promotional item with rate=0 extracts free_quantity=qty and paid quantity=0
    """
    header_elements = [
        {"text": "Description", "box": [[100, 100], [350, 100], [350, 130], [100, 130]]},
        {"text": "Qty", "box": [[400, 100], [480, 100], [480, 130], [400, 130]]},
        {"text": "Rate", "box": [[500, 100], [580, 100], [580, 130], [500, 130]]},
        {"text": "CGST %", "box": [[600, 100], [670, 100], [670, 130], [600, 130]]},
        {"text": "SGST %", "box": [[690, 100], [760, 100], [760, 130], [690, 130]]},
        {"text": "Amount", "box": [[780, 100], [880, 100], [880, 130], [780, 130]]},
    ]

    # Row 1: Compound qty column '20+ 2'
    row1 = [
        {"text": "FLUDAC 20MG 15 CAPS", "box": [[100, 150], [350, 150], [350, 180], [100, 180]]},
        {"text": "20+ 2", "box": [[400, 150], [480, 150], [480, 180], [400, 180]]},
        {"text": "48.71", "box": [[500, 150], [580, 150], [580, 180], [500, 180]]},
        {"text": "2.5", "box": [[600, 150], [670, 150], [670, 180], [600, 180]]},
        {"text": "2.5", "box": [[690, 150], [760, 150], [760, 180], [690, 180]]},
        {"text": "974.20", "box": [[780, 150], [880, 150], [880, 180], [780, 180]]},
    ]

    # Row 2: Free promotional item (rate 0, amount 0)
    row2 = [
        {"text": "PROMO SAMPLE PEN FREE", "box": [[100, 200], [350, 200], [350, 230], [100, 230]]},
        {"text": "5", "box": [[400, 200], [480, 200], [480, 230], [400, 230]]},
        {"text": "0.00", "box": [[500, 200], [580, 200], [580, 230], [500, 230]]},
        {"text": "0.0", "box": [[600, 200], [670, 200], [670, 230], [600, 230]]},
        {"text": "0.0", "box": [[690, 200], [760, 200], [760, 230], [690, 230]]},
        {"text": "0.00", "box": [[780, 200], [880, 200], [880, 230], [780, 230]]},
    ]

    # Footer
    footer = [
        {"text": "Total", "box": [[700, 260], [770, 260], [770, 290], [700, 290]]},
        {"text": "974.20", "box": [[780, 260], [880, 260], [880, 290], [780, 290]]},
    ]

    extractor = TableExtractor()
    res = extractor.extract_table(header_elements + row1 + row2 + footer)

    assert res["status"] == "success"
    assert res["total_items"] == 2

    it1 = res["line_items"][0]
    assert it1["quantity"] == 20.0
    assert it1["free_quantity"] == 2.0
    assert it1["total_quantity"] == 22.0
    assert it1["cgst_amount"] == 24.36
    assert it1["sgst_amount"] == 24.36

    it2 = res["line_items"][1]
    assert it2["quantity"] == 0.0
    assert it2["free_quantity"] == 5.0
    assert it2["total_quantity"] == 5.0


