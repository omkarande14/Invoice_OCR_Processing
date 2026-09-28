# app/input/input_handler.py

from pathlib import Path
from app.input.file_validator import validate_file




def process_folder(folder_path: str) -> dict:
    """
    Runs validate_file() on every file inside folder_path.
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        return {
            "total_files": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "results": [],
            "error": f"Folder not found: {folder_path}",
        }

    results = []

    for item in sorted(folder.iterdir()):
        if item.is_file() and not item.name.startswith("."):
            result = validate_file(str(item))
            results.append(result)

    valid_count = sum(1 for r in results if r["status"] == "valid")
    invalid_count = sum(1 for r in results if r["status"] == "invalid")

    # New: breakdown by format, only counting valid files
    format_counts = {}
    total_pdf_pages = 0
    for r in results:
        if r["status"] == "valid":
            fmt = r["format"]
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            if fmt == "pdf":
                total_pdf_pages += r["page_count"]

    return {
        "total_files": len(results),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "format_breakdown": format_counts,
        "total_pdf_pages": total_pdf_pages,
        "results": results,
    }