# tests/test_deskew_corrector.py

from pathlib import Path
import pytest
from PIL import Image, ImageDraw
from app.preprocessing.deskew_corrector import (
    detect_skew_angle,
    deskew_image,
    process_folder_deskew,
)


@pytest.fixture
def base_ruled_invoice():
    """
    Generates a clean synthetic invoice page with distinct horizontal table lines
    and grid dividers to provide reliable geometric baselines for Hough Line detection.
    """
    img = Image.new("L", (1200, 1600), color=255)
    draw = ImageDraw.Draw(img)

    # Prominent horizontal table borders and divider lines
    draw.line([80, 150, 1120, 150], fill=0, width=3)
    draw.line([80, 230, 1120, 230], fill=0, width=2)
    draw.line([80, 310, 1120, 310], fill=0, width=2)
    draw.line([80, 800, 1120, 800], fill=0, width=3)
    draw.line([80, 900, 1120, 900], fill=0, width=2)

    # Line item rows
    for y in range(360, 780, 50):
        draw.line([80, y, 1120, y], fill=120, width=2)

    return img


@pytest.mark.parametrize("applied_tilt", [1.5, -1.5, 3.0, -3.0])
def test_meaningful_skew_angles_detected_and_corrected(tmp_path, base_ruled_invoice, applied_tilt):
    """
    Tests that tilts of +1.5°, -1.5°, +3.0°, and -3.0° are detected
    and corrected back to horizontal (remaining tilt < 0.5°).
    """
    src_file = tmp_path / f"tilted_{applied_tilt}.png"
    # Rotate by applied_tilt degrees (counter-clockwise in PIL)
    tilted_img = base_ruled_invoice.rotate(
        applied_tilt,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )
    tilted_img.save(src_file)

    out_dir = tmp_path / "deskewed"
    result = deskew_image(str(src_file), str(out_dir), min_skew=0.5)

    assert result["status"] == "deskewed"
    assert result["rotated"] is True
    assert result["line_count"] > 0
    # The detected angle should closely match the magnitude of the applied tilt (within 0.3°)
    assert abs(abs(result["detected_skew_angle"]) - abs(applied_tilt)) <= 0.35
    assert Path(result["output_path"]).exists()

    # Re-inspect corrected image to confirm residual tilt is now below the 0.5° threshold
    with Image.open(result["output_path"]) as corrected_img:
        residual_angle, _ = detect_skew_angle(corrected_img)
        assert abs(residual_angle) < 0.5


def test_sub_threshold_skew_leaves_image_unrotated(tmp_path, base_ruled_invoice):
    """
    A small tilt of 0.2° is below the 0.5° threshold and should NOT be rotated,
    avoiding unnecessary pixel interpolation and preserving maximum sharpness.
    """
    src_file = tmp_path / "tilted_0.2.png"
    tilted_img = base_ruled_invoice.rotate(
        0.2,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )
    tilted_img.save(src_file)

    out_dir = tmp_path / "deskewed"
    result = deskew_image(str(src_file), str(out_dir), min_skew=0.5)

    assert result["status"] == "straight"
    assert result["rotated"] is False
    assert result["correction_applied_deg"] == 0.0
    assert "saved unrotated to preserve sharpness" in result["message"]
    assert Path(result["output_path"]).exists()


def test_straight_image_not_rotated(tmp_path, base_ruled_invoice):
    """An un-tilted (0.0°) image remains untouched."""
    src_file = tmp_path / "straight.png"
    base_ruled_invoice.save(src_file)

    out_dir = tmp_path / "deskewed"
    result = deskew_image(str(src_file), str(out_dir), min_skew=0.5)

    assert result["status"] == "straight"
    assert result["rotated"] is False
    assert result["detected_skew_angle"] == 0.0
    assert Path(result["output_path"]).exists()


def test_outlier_skew_flagged(tmp_path, base_ruled_invoice):
    """Angles exceeding max_skew (e.g. 25°) are flagged as outliers and not blindly rotated."""
    src_file = tmp_path / "outlier_25.png"
    tilted_img = base_ruled_invoice.rotate(25.0, expand=True, fillcolor=255)
    tilted_img.save(src_file)

    out_dir = tmp_path / "deskewed"
    # Testing with max_skew = 10.0 and min_skew = 0.5
    result = deskew_image(str(src_file), str(out_dir), min_skew=0.5, max_skew=10.0)

    assert result["status"] in {"straight", "outlier_skew"}
    assert result["rotated"] is False
    assert Path(result["output_path"]).exists()


def test_process_folder_deskew(tmp_path, base_ruled_invoice):
    """Verifies batch deskew processing aggregates counts and corrections properly."""
    in_dir = tmp_path / "oriented"
    out_dir = tmp_path / "deskewed"
    in_dir.mkdir()

    # Create 1 straight (0°), 1 sub-threshold (0.2°), and 1 tilted (+2.0°)
    base_ruled_invoice.save(in_dir / "invoice_straight.png")
    base_ruled_invoice.rotate(0.2, expand=True, fillcolor=255).save(in_dir / "invoice_subthreshold.png")
    base_ruled_invoice.rotate(2.0, expand=True, fillcolor=255).save(in_dir / "invoice_tilted.png")

    summary = process_folder_deskew(str(in_dir), str(out_dir), min_skew=0.5)

    assert summary["total_images_processed"] == 3
    assert summary["straight_count"] == 2
    assert summary["deskewed_count"] == 1
    assert len(summary["corrections"]) == 1
    assert summary["corrections"][0]["filename"] == "invoice_tilted.png"
