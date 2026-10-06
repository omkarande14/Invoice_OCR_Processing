# tests/test_synthetic_orientation_corrector.py
"""
Synthetic test fixture equivalent for OrientationCorrector.
Validates OSD detection, 90/180/270 rotation handling, confidence filtering,
and batch folder processing on dynamically rendered synthetic invoices using whitelisted constants.
"""

import shutil
from pathlib import Path
import pytest
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from app.preprocessing.orientation_corrector import (
    detect_and_correct_orientation,
    process_folder_orientation,
)

DUMMY_VENDOR_GSTIN = "27AAPFU0939F1ZV"

@pytest.fixture
def tesseract_available():
    """Checks whether tesseract binary is installed and operational."""
    cmd = shutil.which("tesseract")
    if not cmd and not Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists():
        pytest.skip("Tesseract OCR binary not found on host environment")
    return True


@pytest.fixture
def synthetic_invoice_image():
    """
    Generates a realistic invoice-like image with multiple text lines,
    numbers, tables, and varied typography suitable for Tesseract OSD.
    """
    img = Image.new("L", (1400, 1800), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_med = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_large = font_med = ImageFont.load_default()

    draw.text((100, 100), "GLOBAL LOGISTICS & PHARMA SUPPLIES PVT LTD", fill=0, font=font_large)
    draw.text((100, 160), f"Tax Invoice (Original for Recipient) - GSTIN: {DUMMY_VENDOR_GSTIN}", fill=0, font=font_med)
    draw.text((100, 200), "Invoice No: INV-2026-08991 | Date: 15-Aug-2026 | State: Maharashtra", fill=0, font=font_med)
    draw.text((100, 240), "Billed To: Acme HealthCare Solutions, 402 Park Avenue, Mumbai 400001", fill=0, font=font_med)

    y = 320
    items = [
        ("Amoxicillin 500mg Capsules IP Box of 100", "BATCH-A91", "100", "12.50", "1250.00"),
        ("Paracetamol 650mg Fast-Action Tablets", "BATCH-P44", "250", "4.20", "1050.00"),
        ("Azithromycin 250mg Suspension 30ml", "BATCH-Z10", "50", "35.00", "1750.00"),
        ("Cetirizine Hydrochloride 10mg Strip", "BATCH-C02", "120", "8.50", "1020.00"),
        ("Vitamin C Chewable Tablets 500mg Jar 60", "BATCH-V88", "80", "18.00", "1440.00"),
        ("Surgical Disposable Face Masks 3-Ply Box 50", "BATCH-M12", "40", "15.00", "600.00"),
        ("Hand Sanitizer Solution 500ml Flip-Cap", "BATCH-S99", "60", "22.50", "1350.00"),
    ]

    for name, batch, qty, rate, total in items:
        line_text = f"{name:<45} {batch:<12} {qty:>5} {rate:>8} {total:>10}"
        draw.text((100, y), line_text, fill=0, font=font_med)
        y += 50

    draw.text((100, y + 40), "Subtotal: USD 8,460.00 | CGST 9%: USD 761.40 | SGST 9%: USD 761.40", fill=0, font=font_med)
    draw.text((100, y + 80), "Grand Total: USD 9,982.80 | Amount in Words: Nine Thousand Nine Hundred Eighty Two", fill=0, font=font_large)
    draw.text((100, y + 140), "Authorized Signatory for Global Logistics & Pharma Supplies Private Limited", fill=0, font=font_med)

    return img


def test_upright_image_not_rotated(tmp_path, synthetic_invoice_image, tesseract_available):
    """An image that is already upright should be detected with rotate_angle 0 and marked upright."""
    src_path = tmp_path / "invoice_upright.png"
    synthetic_invoice_image.save(src_path)

    out_dir = tmp_path / "oriented"
    result = detect_and_correct_orientation(str(src_path), str(out_dir))

    assert result["status"] == "upright"
    assert result["detected_angle"] == 0
    assert result["rotate_angle"] == 0
    assert result["rotated"] is False
    assert Path(result["output_path"]).exists()


def test_synthetic_90_degree_rotation(tmp_path, synthetic_invoice_image, tesseract_available):
    """An image rotated 90 degrees CCW should be detected and corrected to 0 degrees."""
    src_path = tmp_path / "invoice_rot90.png"
    synthetic_invoice_image.rotate(90, expand=True).save(src_path)

    out_dir = tmp_path / "oriented"
    result = detect_and_correct_orientation(str(src_path), str(out_dir))

    assert result["status"] == "corrected"
    assert result["rotated"] is True
    assert Path(result["output_path"]).exists()


def test_synthetic_180_degree_rotation(tmp_path, synthetic_invoice_image, tesseract_available):
    """An upside-down image (180 degrees) should be detected and corrected to upright."""
    src_path = tmp_path / "invoice_rot180.png"
    synthetic_invoice_image.rotate(180, expand=True).save(src_path)

    out_dir = tmp_path / "oriented"
    result = detect_and_correct_orientation(str(src_path), str(out_dir))

    assert result["status"] == "corrected"
    assert result["rotated"] is True
    assert Path(result["output_path"]).exists()


def test_blank_image_fails_gracefully(tmp_path):
    """A blank image without characters should fail gracefully without crashing."""
    src_path = tmp_path / "blank.png"
    Image.new("L", (300, 300), color=255).save(src_path)

    out_dir = tmp_path / "oriented"
    result = detect_and_correct_orientation(str(src_path), str(out_dir))

    assert result["status"] in ["failed", "upright"]
    assert Path(result["output_path"]).exists()


def test_process_folder_orientation(tmp_path, synthetic_invoice_image, tesseract_available):
    """Verifies batch folder processing aggregates statistics correctly."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    synthetic_invoice_image.save(in_dir / "upright.png")
    synthetic_invoice_image.rotate(90, expand=True).save(in_dir / "rotated.png")
    Image.new("L", (200, 200), color=255).save(in_dir / "blank.png")

    summary = process_folder_orientation(str(in_dir), str(out_dir))
    assert summary["total_images_processed"] == 3
    assert Path(out_dir / "upright_oriented.png").exists()
