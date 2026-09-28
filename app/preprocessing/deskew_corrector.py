# app/preprocessing/deskew_corrector.py

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def detect_skew_angle(image: Image.Image) -> tuple[float, int]:
    """
    Detects the tilt/skew angle (in degrees) of an image using the Hough Line Transform.
    Finds prominent straight horizontal lines (table borders, invoice dividers, text baselines)
    and computes their median angle relative to the horizontal axis.

    Returns:
      (skew_angle, line_count):
        skew_angle: Detected angle in degrees.
                    In image coordinate space, rotating a Pillow image by
                    skew_angle (counter-clockwise) levels the line to horizontal (0°).
        line_count: Number of horizontal lines found and used for the median angle.
    """
    np_img = np.array(image)
    if len(np_img.shape) == 3:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Mild Gaussian blur to smooth pixel staircases from slight rotations
    blurred = cv2.GaussianBlur(np_img, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # Lines should have meaningful length (at least 80px or 8% of width)
    min_line_len = max(80, int(np_img.shape[1] * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_len,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0, 0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        # In image coordinates, y increases downward.
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize angles to [-45, 45] range relative to horizontal
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # Keep lines within +/- 15 degrees of horizontal (ignore vertical table dividers)
        if abs(angle) <= 15.0:
            angles.append(angle)

    if not angles:
        return 0.0, 0

    median_angle = float(np.median(angles))
    return round(median_angle, 2), len(angles)


def deskew_image(
    image_path: str,
    output_folder: str,
    min_skew: float = 0.5,
    max_skew: float = 15.0,
) -> dict:
    """
    Detects skew on a single image and applies fine-angle rotation if skew exceeds min_skew.

    Traceability:
      The source image is NEVER modified or overwritten.
      Corrected output is saved to: <output_folder>/<stem>_deskewed.png

    Parameters:
      image_path: Path to the input image (typically in data/oriented/).
      output_folder: Destination folder (typically data/deskewed/).
      min_skew: Minimum skew in degrees below which no rotation is performed (avoids interpolation blur).
      max_skew: Maximum plausible skew angle (larger angles are flagged as anomalies).

    Returns:
      A dictionary detailing the operation:
        - filename: original file name
        - output_path: path to saved deskewed image
        - detected_skew_angle: detected angle in degrees
        - correction_applied_deg: degrees of rotation applied (0.0 if not rotated)
        - line_count: number of horizontal lines detected
        - rotated: boolean indicating if rotation occurred
        - status: 'straight', 'deskewed', 'outlier_skew', or 'failed'
        - message: human-readable explanation
    """
    path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{path.stem}_deskewed.png"
    output_path = out_dir / output_filename

    result = {
        "filename": path.name,
        "output_path": str(output_path),
        "detected_skew_angle": 0.0,
        "correction_applied_deg": 0.0,
        "line_count": 0,
        "rotated": False,
        "status": "unknown",
        "message": "",
    }

    try:
        with Image.open(path) as img:
            skew_angle, line_count = detect_skew_angle(img)
            result["detected_skew_angle"] = skew_angle
            result["line_count"] = line_count

            abs_skew = abs(skew_angle)

            if abs_skew < min_skew:
                # Sub-threshold tilt: image is straight enough. Preserve pristine sharpness.
                img.save(output_path, format="PNG")
                result["status"] = "straight"
                result["rotated"] = False
                result["message"] = (
                    f"Skew {skew_angle:+.2f}° is below threshold ({min_skew}°); saved unrotated to preserve sharpness"
                )
            elif abs_skew > max_skew:
                # Outlier: unusually large tilt beyond normal scanner skew. Flag for review.
                img.save(output_path, format="PNG")
                result["status"] = "outlier_skew"
                result["rotated"] = False
                result["message"] = (
                    f"Skew {skew_angle:+.2f}° exceeds max threshold ({max_skew}°); saved unrotated for review"
                )
            else:
                # Meaningful skew detected: rotate to level horizontal lines.
                # In Pillow, rotate(angle) is counter-clockwise.
                # rotating by skew_angle levels the detected tilt.
                deskewed_img = img.rotate(
                    skew_angle,
                    expand=True,
                    resample=Image.Resampling.BICUBIC,
                    fillcolor=255,  # Fill newly exposed corner margins with white
                )
                deskewed_img.save(output_path, format="PNG")
                result["correction_applied_deg"] = skew_angle
                result["status"] = "deskewed"
                result["rotated"] = True
                result["message"] = (
                    f"Rotated {skew_angle:+.2f}° to level horizontal baselines (from {line_count} lines)"
                )

    except Exception as e:
        result["status"] = "failed"
        result["message"] = f"Deskew processing error: {e}"
        # Fallback copy to ensure pipeline continuity
        try:
            with Image.open(path) as img:
                img.save(output_path, format="PNG")
        except Exception:
            pass

    return result


def process_folder_deskew(
    input_folder: str,
    output_folder: str,
    target_files: list = None,
    min_skew: float = 0.5,
    max_skew: float = 15.0,
) -> dict:
    """
    Processes all PNG images in input_folder (or subset in target_files),
    correcting small tilt angles and saving outputs into output_folder.

    Returns:
      {
        "total_images_processed": int,
        "straight_count": int,
        "deskewed_count": int,
        "outlier_count": int,
        "failed_count": int,
        "threshold_deg": float,
        "corrections": [...],
        "all_results": [...]
      }
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_files is not None:
        target_set = set(target_files)
        image_files = sorted(
            [p for p in in_dir.iterdir() if p.is_file() and p.name in target_set]
        )
    else:
        image_files = sorted(
            [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        )

    summary = {
        "total_images_processed": len(image_files),
        "straight_count": 0,
        "deskewed_count": 0,
        "outlier_count": 0,
        "failed_count": 0,
        "threshold_deg": min_skew,
        "corrections": [],
        "all_results": [],
    }

    for img_path in image_files:
        res = deskew_image(
            str(img_path),
            str(out_dir),
            min_skew=min_skew,
            max_skew=max_skew,
        )
        summary["all_results"].append(res)

        if res["status"] == "straight":
            summary["straight_count"] += 1
        elif res["status"] == "deskewed":
            summary["deskewed_count"] += 1
            summary["corrections"].append(res)
        elif res["status"] == "outlier_skew":
            summary["outlier_count"] += 1
        elif res["status"] == "failed":
            summary["failed_count"] += 1

    return summary
