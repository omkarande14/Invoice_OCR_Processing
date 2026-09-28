# app/input/file_validator.py

from pathlib import Path
import pymupdf as fitz       # PyMuPDF - for reading PDF files
from PIL import Image  # Pillow - for reading image files

# The file types Milestone 1 supports.
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def validate_file(file_path: str) -> dict:
    """
    Checks a single file and reports whether it's a usable invoice input.

    Returns a dictionary like:
      {"status": "valid", "filename": "...", "format": "pdf", "page_count": 3}
      or
      {"status": "invalid", "filename": "...", "reason": "..."}
    """
    path = Path(file_path)
    filename = path.name

    # Step 1: Does the file even exist?
    if not path.exists():
        return {
            "status": "invalid",
            "filename": filename,
            "reason": "File does not exist",
        }

    # Step 2: Is the extension one we support?
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return {
            "status": "invalid",
            "filename": filename,
            "reason": f"Unsupported file format: {extension or 'no extension'}",
        }

    # Step 3: Handle PDFs
    if extension == ".pdf":
        return _validate_pdf(path, filename)

    # Step 4: Handle images (.jpg, .jpeg, .png)
    return _validate_image(path, filename, extension)


def _validate_pdf(path: Path, filename: str) -> dict:
    try:
        doc = fitz.open(path)
    except Exception as e:
        return {
            "status": "invalid",
            "filename": filename,
            "reason": f"Could not open PDF (possibly corrupted): {e}",
        }

    if doc.is_encrypted:
        # Some encrypted PDFs can still be opened with an empty password.
        # If PyMuPDF couldn't authenticate, we treat it as unreadable.
        if not doc.authenticate(""):
            doc.close()
            return {
                "status": "invalid",
                "filename": filename,
                "reason": "PDF is password-protected/encrypted",
            }

    page_count = len(doc)
    doc.close()

    if page_count == 0:
        return {
            "status": "invalid",
            "filename": filename,
            "reason": "PDF has zero pages",
        }

    return {
        "status": "valid",
        "filename": filename,
        "format": "pdf",
        "page_count": page_count,
    }


def _validate_image(path: Path, filename: str, extension: str) -> dict:
    try:
        with Image.open(path) as img:
            img.verify()  # checks the image isn't corrupted, without fully loading it
    except Exception as e:
        return {
            "status": "invalid",
            "filename": filename,
            "reason": f"Could not open image (possibly corrupted): {e}",
        }

    return {
        "status": "valid",
        "filename": filename,
        "format": extension.lstrip("."),
        "page_count": 1,
    }