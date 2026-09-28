# tests/test_template_similarity.py

import io
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
from app.preprocessing.template_similarity_detector import (
    compute_perceptual_hash,
    find_template_similar_images,
)


@pytest.fixture
def base_invoice_image():
    """Generates a synthetic grayscale invoice-like image with header, boxes, and table lines."""
    img = Image.new("L", (400, 500), 255)
    draw = ImageDraw.Draw(img)
    # Header logo / box
    draw.rectangle([30, 20, 370, 70], fill=120)
    # Metadata box
    draw.rectangle([30, 90, 370, 160], outline=40, width=2)
    # Line items table
    draw.rectangle([30, 180, 370, 420], outline=40, width=2)
    for y in range(220, 420, 40):
        draw.line([30, y, 370, y], fill=100, width=1)
    for x in [100, 200, 300]:
        draw.line([x, 180, x, 420], fill=100, width=1)
    return img


def test_recompressed_jpeg_detected_as_template_similar(tmp_path, base_invoice_image):
    """Images sharing layout despite compression artifacts have distance <= 4."""
    p1 = tmp_path / "invoice_original.png"
    base_invoice_image.save(p1)

    buf = io.BytesIO()
    base_invoice_image.save(buf, format="JPEG", quality=25)
    buf.seek(0)
    img_compressed = Image.open(buf)
    p2 = tmp_path / "invoice_recompressed.png"
    img_compressed.save(p2)

    result = find_template_similar_images(str(tmp_path), threshold=4)

    assert result["total_images_analyzed"] == 2
    assert result["total_template_similarity_pairs"] == 1
    pair = result["template_similarity_pairs"][0]
    assert pair["hamming_distance"] <= 4
    assert set([pair["file_1"], pair["file_2"]]) == {"invoice_original.png", "invoice_recompressed.png"}
    assert len(result["template_similarity_groups"]) == 1
    assert "NOT duplicate content" in result["note"]


def test_minor_crop_detected_as_template_similar(tmp_path, base_invoice_image):
    """A slightly cropped image sharing layout has distance <= 4."""
    p1 = tmp_path / "invoice_original.png"
    base_invoice_image.save(p1)

    w, h = base_invoice_image.size
    crop_x = 2
    crop_y = 2
    img_cropped = base_invoice_image.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    p2 = tmp_path / "invoice_cropped.png"
    img_cropped.save(p2)

    result = find_template_similar_images(str(tmp_path), threshold=4)

    assert result["total_images_analyzed"] == 2
    assert result["total_template_similarity_pairs"] == 1
    assert result["template_similarity_pairs"][0]["hamming_distance"] <= 4


def test_borderline_similarity_pair_logged_separately(tmp_path, base_invoice_image):
    """Pairs with distance between 5 and 8 should be logged in borderline_pairs, not template_similarity_pairs."""
    p1 = tmp_path / "invoice_original.png"
    base_invoice_image.save(p1)

    w, h = base_invoice_image.size
    img_borderline = base_invoice_image.crop((5, 6, w - 5, h - 6))
    p2 = tmp_path / "invoice_borderline.png"
    img_borderline.save(p2)

    result = find_template_similar_images(str(tmp_path), threshold=4, borderline_min=5, borderline_max=8)

    assert result["total_images_analyzed"] == 2
    assert result["total_template_similarity_pairs"] == 0
    assert result["total_borderline_pairs"] == 1
    borderline_pair = result["borderline_pairs"][0]
    assert 5 <= borderline_pair["hamming_distance"] <= 8


def test_different_images_not_flagged(tmp_path, base_invoice_image):
    """Completely distinct visual layouts should not be flagged as template-similar or borderline."""
    p1 = tmp_path / "invoice.png"
    base_invoice_image.save(p1)

    img_diff = Image.new("L", (400, 500), 255)
    draw = ImageDraw.Draw(img_diff)
    draw.ellipse([50, 50, 350, 450], fill=30)
    p2 = tmp_path / "circle_document.png"
    img_diff.save(p2)

    result = find_template_similar_images(str(tmp_path), threshold=4, borderline_min=5, borderline_max=8)

    assert result["total_images_analyzed"] == 2
    assert result["total_template_similarity_pairs"] == 0
    assert result["total_borderline_pairs"] == 0


def test_target_files_filter(tmp_path, base_invoice_image):
    """When target_files is specified, only the specified subset is evaluated."""
    p1 = tmp_path / "inv1.png"
    p2 = tmp_path / "inv2.png"
    p3 = tmp_path / "inv3.png"

    base_invoice_image.save(p1)
    base_invoice_image.save(p2)
    base_invoice_image.save(p3)

    result = find_template_similar_images(str(tmp_path), target_files=["inv1.png", "inv2.png"], threshold=4)

    assert result["total_images_analyzed"] == 2
    assert result["total_template_similarity_pairs"] == 1
    assert result["template_similarity_pairs"][0]["file_1"] == "inv1.png"
    assert result["template_similarity_pairs"][0]["file_2"] == "inv2.png"


def test_invalid_folder():
    """Non-existent directory should return an error structure safely without raising exceptions."""
    result = find_template_similar_images("non_existent_folder_path")
    assert "error" in result
    assert result["total_images_analyzed"] == 0
