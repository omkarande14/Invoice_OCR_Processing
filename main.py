# main.py

import json
from datetime import datetime
from pathlib import Path
from app.input.input_handler import process_folder
from app.preprocessing.image_converter import convert_file
from app.preprocessing.duplicate_detector import find_exact_duplicates
from app.preprocessing.template_similarity_detector import find_template_similar_images
from app.preprocessing.orientation_corrector import process_folder_orientation
from app.preprocessing.deskew_corrector import process_folder_deskew

INPUT_FOLDER = "data/input"
VALIDATION_FOLDER = "data/validation"
PROCESSED_FOLDER = "data/processed"
ORIENTED_FOLDER = "data/oriented"
DESKEWED_FOLDER = "data/deskewed"


def main():
    print(f"Scanning folder: {INPUT_FOLDER}\n")

    summary = process_folder(INPUT_FOLDER)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"Total files found : {summary['total_files']}")
    print(f"Valid files       : {summary['valid_count']}")
    print(f"Invalid files     : {summary['invalid_count']}")
    print(f"Format breakdown  : {summary['format_breakdown']}")
    print(f"Total PDF pages   : {summary['total_pdf_pages']}\n")

    if summary["invalid_count"] > 0:
        print("Invalid files and reasons:")
        for r in summary["results"]:
            if r["status"] == "invalid":
                print(f"  - {r['filename']}: {r['reason']}")
        print()

    # --- Milestone 2: convert every valid file into images ---
    print(f"Converting valid files to images in: {PROCESSED_FOLDER}\n")
    total_images_created = 0
    conversion_errors = []

    for r in summary["results"]:
        if r["status"] == "valid":
            source_path = Path(INPUT_FOLDER) / r["filename"]
            try:
                saved_images = convert_file(str(source_path), PROCESSED_FOLDER)
                total_images_created += len(saved_images)
            except Exception as e:
                conversion_errors.append({"filename": r["filename"], "error": str(e)})

    print(f"Total images created : {total_images_created}")
    print(f"Conversion errors     : {len(conversion_errors)}\n")

    if conversion_errors:
        print("Files that failed conversion:")
        for e in conversion_errors:
            print(f"  - {e['filename']}: {e['error']}")
        print()

    summary["total_images_created"] = total_images_created
    summary["conversion_errors"] = conversion_errors

    # --- Milestone 3a: exact duplicate detection ---
    print("Checking for exact duplicate images...\n")
    duplicate_summary = find_exact_duplicates(PROCESSED_FOLDER)

    print(f"Total images         : {duplicate_summary['total_images']}")
    print(f"Unique images        : {duplicate_summary['unique_images']}")
    print(f"Duplicate groups     : {duplicate_summary['total_duplicate_groups']}")
    print(f"Redundant copies     : {duplicate_summary['redundant_copies']}\n")

    if duplicate_summary["duplicate_groups"]:
        print("Duplicate groups found:")
        for i, group in enumerate(duplicate_summary["duplicate_groups"], start=1):
            print(f"  Group {i}: {group}")
        print()

    summary["duplicate_detection"] = duplicate_summary

    # --- Milestone 3b: template similarity detection (shared layout) ---
    print("Analyzing template similarity across unique images (shared layout detection)...\n")
    redundant_exact_files = set()
    for group in duplicate_summary.get("duplicate_groups", []):
        redundant_exact_files.update(group[1:])

    all_processed_pngs = sorted([p.name for p in Path(PROCESSED_FOLDER).glob("*.png")])
    unique_images_to_check = [f for f in all_processed_pngs if f not in redundant_exact_files]

    template_sim_summary = find_template_similar_images(
        PROCESSED_FOLDER,
        target_files=unique_images_to_check,
        threshold=4,
        borderline_min=5,
        borderline_max=8,
    )

    print(f"Images analyzed            : {template_sim_summary['total_images_analyzed']}")
    print(f"Template-similarity pairs  : {template_sim_summary['total_template_similarity_pairs']} (threshold <= {template_sim_summary['threshold']})")
    print(f"Template-similarity groups : {template_sim_summary['total_template_similarity_groups']}")
    print(f"Borderline layout pairs    : {template_sim_summary['total_borderline_pairs']} (distance 5-8)\n")

    if template_sim_summary["template_similarity_pairs"]:
        print("Template-similar pairs found (same vendor layout, distinct content):")
        for pair in template_sim_summary["template_similarity_pairs"]:
            print(f"  - {pair['file_1']} <-> {pair['file_2']} (Hamming dist: {pair['hamming_distance']}, similarity: {pair['similarity_percentage']}%)")
        print()

    if template_sim_summary["borderline_pairs"]:
        print("Borderline layout pairs (distance 5-8, watchlist only):")
        for pair in template_sim_summary["borderline_pairs"]:
            print(f"  - {pair['file_1']} <-> {pair['file_2']} (Hamming dist: {pair['hamming_distance']}, similarity: {pair['similarity_percentage']}%)")
        print()

    summary["template_similarity"] = template_sim_summary

    # --- Milestone 3c: orientation detection and correction ---
    print(f"Detecting and correcting image orientation in: {ORIENTED_FOLDER}...\n")
    orientation_summary = process_folder_orientation(
        PROCESSED_FOLDER,
        ORIENTED_FOLDER,
        target_files=unique_images_to_check,
        min_confidence=2.0,
    )

    print(f"Total unique images analyzed : {orientation_summary['total_images_processed']}")
    print(f"Already upright (0 deg)      : {orientation_summary['upright_count']}")
    print(f"Rotated to upright           : {orientation_summary['corrected_count']}")
    print(f"Low confidence (flagged)     : {orientation_summary['low_confidence_count']}")
    print(f"Failed / unreadable          : {orientation_summary['failed_count']}\n")

    if orientation_summary["corrections"]:
        print("Rotated images:")
        for c in orientation_summary["corrections"]:
            print(f"  - {c['filename']}: rotated {c['rotate_angle']} deg clockwise (confidence: {c['confidence']:.2f}) -> {c['output_path']}")
        print()

    if orientation_summary["low_confidence_detections"]:
        print("Flagged low-confidence detections (review recommended):")
        for lc in orientation_summary["low_confidence_detections"]:
            print(f"  - {lc['filename']}: suggested {lc['rotate_angle']} deg, confidence {lc['confidence']:.2f} (saved unrotated)")
        print()

    if orientation_summary["failed_detections"]:
        print("Failed detections (zero/insufficient text):")
        for fd in orientation_summary["failed_detections"]:
            print(f"  - {fd['filename']}: {fd['message']}")
        print()

    summary["orientation_correction"] = orientation_summary

    # --- Milestone 3d: deskew detection and correction ---
    print(f"Detecting and correcting fine-angle skew in: {DESKEWED_FOLDER}...\n")
    deskew_summary = process_folder_deskew(
        ORIENTED_FOLDER,
        DESKEWED_FOLDER,
        min_skew=0.5,
        max_skew=15.0,
    )

    print(f"Total images checked for skew : {deskew_summary['total_images_processed']}")
    print(f"Already straight (<0.5 deg)   : {deskew_summary['straight_count']}")
    print(f"Deskewed (corrected)          : {deskew_summary['deskewed_count']}")
    print(f"Outlier skews (>15 deg)       : {deskew_summary['outlier_count']}")
    print(f"Failed / unreadable           : {deskew_summary['failed_count']}\n")

    if deskew_summary["corrections"]:
        print("Deskewed images:")
        for dc in deskew_summary["corrections"]:
            print(f"  - {dc['filename']}: rotated {dc['correction_applied_deg']:+.2f} deg (from {dc['line_count']} lines) -> {dc['output_path']}")
        print()

    summary["deskew_correction"] = deskew_summary

    # --- Save the full report (now includes everything above) ---
    Path(VALIDATION_FOLDER).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(VALIDATION_FOLDER) / f"validation_report_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()