# app/preprocessing/orientation_corrector.py

import os
import shutil
from pathlib import Path
from PIL import Image
import pytesseract

# Default standard Tesseract installation location on Windows
DEFAULT_TESSERACT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def configure_tesseract_path() -> str:
    """
    Ensures pytesseract knows where the Tesseract binary is located.
    Checks system PATH first; if not found, checks standard Windows install paths.
    """
    cmd = shutil.which("tesseract")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return cmd

    if os.path.exists(DEFAULT_TESSERACT_WINDOWS_PATH):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_WINDOWS_PATH
        return DEFAULT_TESSERACT_WINDOWS_PATH

    # If neither is found, pytesseract will attempt default command name ('tesseract')
    return "tesseract"


# Initialize tesseract path on module load
configure_tesseract_path()


def score_orientation_keywords(img: Image.Image) -> int:
    """
    Counts matches of standard invoice domain keywords to verify upright readability.
    """
    try:
        txt = pytesseract.image_to_string(img).lower()
        keywords = {
            "invoice", "tax", "gst", "total", "date", "bill", "amount",
            "rate", "qty", "hsn", "mrp", "phone", "pack", "batch",
            "medico", "stores", "hospital", "buyer", "customer", "seller"
        }
        return sum(1 for kw in keywords if kw in txt)
    except Exception:
        return 0


def detect_and_correct_orientation(
    image_path: str,
    output_folder: str,
    min_confidence: float = 0.5,
) -> dict:
    """
    Detects the orientation of an image using Tesseract OSD (Orientation and Script Detection)
    and content keyword verification, rotates it to upright (0 degrees) if needed,
    and saves the result to output_folder.

    Traceability:
      The original image in image_path is NEVER modified or overwritten.
      The output is saved as: <output_folder>/<image_name_stem>_oriented.png

    Parameters:
      image_path: Path to the source image.
      output_folder: Directory to save the corrected image.
      min_confidence: Minimum OSD confidence required to apply rotation (default 0.5).
                      If confidence is below this threshold, the image is saved without rotation
                      and flagged as low-confidence.

    Returns:
      A dictionary detailing:
        - filename: name of the source image
        - output_path: path to the saved oriented image
        - detected_angle: orientation detected by OSD (0, 90, 180, 270)
        - rotate_angle: clockwise degrees needed to make upright (0, 90, 180, 270)
        - confidence: confidence score reported by Tesseract
        - rotated: boolean indicating if a rotation was performed
        - status: 'upright', 'corrected', 'low_confidence', or 'failed'
        - message: human-readable explanation
    """
    path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{path.stem}_oriented.png"
    output_path = out_dir / output_filename

    result = {
        "filename": path.name,
        "output_path": str(output_path),
        "detected_angle": 0,
        "rotate_angle": 0,
        "confidence": 0.0,
        "rotated": False,
        "status": "unknown",
        "message": "",
    }

    try:
        with Image.open(path) as img:
            # Run Tesseract Orientation and Script Detection
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)

            detected_angle = int(osd.get("orientation", 0))
            rotate_angle = int(osd.get("rotate", 0))
            confidence = float(osd.get("orientation_conf", 0.0))

            result["detected_angle"] = detected_angle
            result["rotate_angle"] = rotate_angle
            result["confidence"] = confidence

            # Check if rotation is needed
            if rotate_angle == 0:
                # Already upright according to OSD
                img.save(output_path, format="PNG")
                result["status"] = "upright"
                result["rotated"] = False
                result["message"] = f"Image is already upright (confidence: {confidence:.2f})"
            elif confidence < min_confidence:
                # Rotation suggested but confidence is too low to trust blindly
                img.save(output_path, format="PNG")
                result["status"] = "low_confidence"
                result["rotated"] = False
                result["message"] = (
                    f"Suggested rotate {rotate_angle}° but confidence {confidence:.2f} "
                    f"is below threshold {min_confidence:.2f}; saved unrotated for review"
                )
            else:
                # Confident non-upright image: rotate to upright
                # In Pillow, rotate(angle) is counter-clockwise.
                # rotate(-rotate_angle) rotates clockwise by rotate_angle.
                corrected_img = img.rotate(-rotate_angle, expand=True)
                corrected_img.save(output_path, format="PNG")
                result["status"] = "corrected"
                result["rotated"] = True
                result["message"] = (
                    f"Rotated {rotate_angle}° clockwise to upright (confidence: {confidence:.2f})"
                )

    except pytesseract.TesseractError as e:
        # Handles images with too few characters, unusual resolution, or blank pages
        # Attempt fallback keyword scoring across cardinal angles (0, 90, 180, 270)
        try:
            with Image.open(path) as img:
                best_angle = 0
                best_score = score_orientation_keywords(img)
                for ang in [90, 180, 270]:
                    cand_img = img.rotate(-ang, expand=True)
                    score = score_orientation_keywords(cand_img)
                    if score > best_score and score >= 3:
                        best_score = score
                        best_angle = ang

                if best_angle != 0:
                    corrected_img = img.rotate(-best_angle, expand=True)
                    corrected_img.save(output_path, format="PNG")
                    result["status"] = "corrected"
                    result["rotated"] = True
                    result["rotate_angle"] = best_angle
                    result["confidence"] = 3.0
                    result["message"] = f"Rotated {best_angle}° clockwise using keyword fallback (score: {best_score})"
                    return result
                else:
                    img.save(output_path, format="PNG")
                    result["status"] = "failed"
                    result["message"] = f"Tesseract OSD could not determine orientation: {e}"
        except Exception:
            shutil.copyfile(str(path), str(output_path))
            result["status"] = "failed"
            result["message"] = f"Tesseract OSD could not determine orientation: {e}"

    except Exception as e:
        result["status"] = "failed"
        result["message"] = f"Unexpected error during orientation correction: {e}"
        try:
            shutil.copyfile(str(path), str(output_path))
        except Exception:
            pass

    return result


def process_folder_orientation(
    input_folder: str,
    output_folder: str,
    target_files: list = None,
    min_confidence: float = 2.0,
) -> dict:
    """
    Processes all PNG images in input_folder (or subset in target_files), corrects
    their orientation, and saves them into output_folder.

    Returns:
      {
        "total_images_processed": int,
        "upright_count": int,
        "corrected_count": int,
        "low_confidence_count": int,
        "failed_count": int,
        "corrections": [...],
        "low_confidence_detections": [...],
        "failed_detections": [...],
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
        "upright_count": 0,
        "corrected_count": 0,
        "low_confidence_count": 0,
        "failed_count": 0,
        "corrections": [],
        "low_confidence_detections": [],
        "failed_detections": [],
        "all_results": [],
    }

    for img_path in image_files:
        res = detect_and_correct_orientation(
            str(img_path),
            str(out_dir),
            min_confidence=min_confidence,
        )

        summary["all_results"].append(res)

        if res["status"] == "upright":
            summary["upright_count"] += 1
        elif res["status"] == "corrected":
            summary["corrected_count"] += 1
            summary["corrections"].append(res)
        elif res["status"] == "low_confidence":
            summary["low_confidence_count"] += 1
            summary["low_confidence_detections"].append(res)
        elif res["status"] == "failed":
            summary["failed_count"] += 1
            summary["failed_detections"].append(res)

    return summary
