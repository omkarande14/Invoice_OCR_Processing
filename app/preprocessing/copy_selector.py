# app/preprocessing/copy_selector.py
"""
Dual-Invoice Page Detector & Customer Copy Selector.

Detects whether an invoice image contains multiple side-by-side or stacked copies
on a single page (e.g. 'Customer Copy' on left, 'Office Copy' on right, or
'Original for Recipient' vs 'Duplicate for Transporter').
Calculates the spatial dividing boundary through the margin/gutter and crops
the page strictly to the primary deliverable copy (Customer Copy / Original).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract


@dataclass
class CopySelectionResult:
    is_dual_copy: bool
    selected_copy: Optional[str]
    detected_labels: List[str]
    split_axis: Optional[str]  # 'horizontal' or 'vertical'
    boundary_coord: Optional[int]
    output_image_path: Path
    details: Dict[str, Any]


class DualCopySelector:
    """
    Detects dual-copy invoice pages and extracts only the Customer/Original copy.
    """

    PRIMARY_LABELS = [
        "customer copy", "original for recipient", "original copy", "original"
    ]
    SECONDARY_LABELS = [
        "office copy", "duplicate for transporter", "duplicate copy",
        "triplicate for supplier", "duplicate", "transporter copy"
    ]

    DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def __init__(self, tesseract_cmd: Optional[str] = None):
        import os
        cmd = tesseract_cmd or self.DEFAULT_TESSERACT_CMD
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd

    def inspect_and_select(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        extracted_lines: Optional[List[Dict[str, Any]]] = None
    ) -> CopySelectionResult:
        """
        Main entrypoint:
        1. Checks if dual-copy anchors exist in the header region.
        2. If detected, determines horizontal or vertical split and finds the dividing gutter.
        3. Crops the image to the Customer Copy and saves it.
        4. If not dual copy, returns the original image path.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
            pil_img_rgb = pil_img.convert("RGB")

        # Step 1: Detect copy anchor text
        anchors = self._detect_anchors(img_path, (w, h), extracted_lines)
        primary_anchor = anchors.get("primary")
        secondary_anchor = anchors.get("secondary")

        if not (primary_anchor and secondary_anchor):
            return CopySelectionResult(
                is_dual_copy=False,
                selected_copy=None,
                detected_labels=[k for k, v in anchors.items() if v],
                split_axis=None,
                boundary_coord=None,
                output_image_path=img_path,
                details={"reason": "No dual-copy anchor pair detected"}
            )

        # Step 2: Determine split axis (horizontal vs vertical)
        p_box = primary_anchor["box"]
        s_box = secondary_anchor["box"]
        p_xc = (min(p[0] for p in p_box) + max(p[0] for p in p_box)) / 2.0
        s_xc = (min(p[0] for p in s_box) + max(p[0] for p in s_box)) / 2.0
        p_yc = (min(p[1] for p in p_box) + max(p[1] for p in p_box)) / 2.0
        s_yc = (min(p[1] for p in s_box) + max(p[1] for p in s_box)) / 2.0

        is_horizontal_split = abs(p_xc - s_xc) > abs(p_yc - s_yc)

        out_folder = output_dir if output_dir else img_path.parent
        out_folder.mkdir(parents=True, exist_ok=True)

        if is_horizontal_split:
            # Customer copy is typically on the left half (lower X)
            split_x = self._find_vertical_gutter(pil_img_rgb, (w, h), p_xc, s_xc)
            
            # Crop to Customer Copy (left half: 0 to split_x)
            cropped = pil_img_rgb.crop((0, 0, split_x, h))
            out_filename = f"{img_path.stem}_customer_copy.png"
            out_path = out_folder / out_filename
            cropped.save(out_path)

            return CopySelectionResult(
                is_dual_copy=True,
                selected_copy="Customer Copy",
                detected_labels=[primary_anchor["text"], secondary_anchor["text"]],
                split_axis="horizontal",
                boundary_coord=split_x,
                output_image_path=out_path,
                details={
                    "original_size": (w, h),
                    "cropped_size": cropped.size,
                    "split_x": split_x,
                    "primary_label": primary_anchor["text"],
                    "secondary_label": secondary_anchor["text"]
                }
            )
        else:
            # Vertical split (top vs bottom)
            split_y = self._find_horizontal_gutter(pil_img_rgb, (w, h), p_yc, s_yc)
            cropped = pil_img_rgb.crop((0, 0, w, split_y))
            out_filename = f"{img_path.stem}_customer_copy.png"
            out_path = out_folder / out_filename
            cropped.save(out_path)

            return CopySelectionResult(
                is_dual_copy=True,
                selected_copy="Customer Copy",
                detected_labels=[primary_anchor["text"], secondary_anchor["text"]],
                split_axis="vertical",
                boundary_coord=split_y,
                output_image_path=out_path,
                details={
                    "original_size": (w, h),
                    "cropped_size": cropped.size,
                    "split_y": split_y,
                    "primary_label": primary_anchor["text"],
                    "secondary_label": secondary_anchor["text"]
                }
            )

    def _detect_anchors(
        self,
        image_path: Path,
        dimensions: Tuple[int, int],
        extracted_lines: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Detects presence and bounding boxes of primary and secondary copy labels."""
        w, h = dimensions
        anchors = {"primary": None, "secondary": None}

        # Use existing OCR lines if provided
        lines = extracted_lines
        if not lines:
            # Fast lightweight OCR on the top 25% header strip
            with Image.open(image_path) as im:
                header_strip = im.crop((0, 0, w, int(h * 0.25)))
            
            try:
                ocr_data = pytesseract.image_to_data(header_strip, output_type=pytesseract.Output.DICT)
            except Exception:
                return anchors

            word_list = []
            n_boxes = len(ocr_data["text"])
            for i in range(n_boxes):
                t = ocr_data["text"][i].strip()
                if t:
                    x, y, bw, bh = (
                        ocr_data["left"][i],
                        ocr_data["top"][i],
                        ocr_data["width"][i],
                        ocr_data["height"][i]
                    )
                    word_list.append({
                        "text": t,
                        "line_id": (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i]),
                        "box": [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
                    })

            lines = []
            # 1. Single words
            for w_item in word_list:
                lines.append(w_item)

            # 2. Sliding bigrams (e.g. 'Customer Copy', 'Office Copy')
            for i in range(len(word_list) - 1):
                w1 = word_list[i]
                w2 = word_list[i + 1]
                if w1["line_id"] == w2["line_id"]:
                    bigram_text = f"{w1['text']} {w2['text']}"
                    b_min_x = min(w1["box"][0][0], w2["box"][0][0])
                    b_min_y = min(w1["box"][0][1], w2["box"][0][1])
                    b_max_x = max(w1["box"][2][0], w2["box"][2][0])
                    b_max_y = max(w1["box"][2][1], w2["box"][2][1])
                    lines.append({
                        "text": bigram_text,
                        "box": [[b_min_x, b_min_y], [b_max_x, b_min_y], [b_max_x, b_max_y], [b_min_x, b_max_y]]
                    })

        # Match labels against lines
        for el in lines:
            t = el["text"].strip().lower()
            b = el["box"]
            ymin = min(p[1] for p in b)
            # Confine search to top 25% of image
            if ymin > h * 0.25:
                continue

            if not anchors["primary"]:
                for pl in self.PRIMARY_LABELS:
                    if pl in t:
                        anchors["primary"] = el
                        break

            if not anchors["secondary"]:
                for sl in self.SECONDARY_LABELS:
                    if sl in t:
                        anchors["secondary"] = el
                        break

        return anchors

    def _find_vertical_gutter(
        self,
        image: Image.Image,
        dimensions: Tuple[int, int],
        p_xc: float,
        s_xc: float
    ) -> int:
        """Finds the optimal vertical dividing column (white gutter or border line) between copies."""
        w, h = dimensions
        arr = np.array(image.convert("L"))

        # Search region: strictly between primary anchor and secondary anchor
        left_bound = int(max(p_xc + 100, w * 0.40))
        right_bound = int(min(s_xc - 100, w * 0.75))

        if right_bound <= left_bound:
            return int(w / 2.0)

        # Profile: column average brightness across Y in the middle 70% of the page
        sub = arr[int(h * 0.15):int(h * 0.85), left_bound:right_bound]
        col_means = sub.mean(axis=0)

        # Find continuous white run (intensity > 245)
        is_white = col_means > 245
        max_run = 0
        best_start = 0
        cur_run = 0
        cur_start = 0

        for i, val in enumerate(is_white):
            if val:
                if cur_run == 0:
                    cur_start = i
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
                    best_start = cur_start
            else:
                cur_run = 0

        if max_run >= 10:
            # Centered in the white gutter
            return left_bound + best_start + (max_run // 2)

        # Fallback: find darkest column (vertical divider line) or midpoint
        darkest_idx = int(np.argmin(col_means))
        return left_bound + darkest_idx

    def _find_horizontal_gutter(
        self,
        image: Image.Image,
        dimensions: Tuple[int, int],
        p_yc: float,
        s_yc: float
    ) -> int:
        """Finds the optimal horizontal dividing row for vertically stacked copies."""
        w, h = dimensions
        arr = np.array(image.convert("L"))
        top_bound = int(max(p_yc + 50, h * 0.35))
        bot_bound = int(min(s_yc - 50, h * 0.65))

        if bot_bound <= top_bound:
            return int(h / 2.0)

        sub = arr[top_bound:bot_bound, int(w * 0.10):int(w * 0.90)]
        row_means = sub.mean(axis=1)

        is_white = row_means > 245
        max_run = 0
        best_start = 0
        cur_run = 0
        cur_start = 0

        for i, val in enumerate(is_white):
            if val:
                if cur_run == 0:
                    cur_start = i
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
                    best_start = cur_start
            else:
                cur_run = 0

        if max_run >= 10:
            return top_bound + best_start + (max_run // 2)

        return (top_bound + bot_bound) // 2
