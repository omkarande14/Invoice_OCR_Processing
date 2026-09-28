# app/preprocessing/image_converter.py

from pathlib import Path
import pymupdf as fitz
from PIL import Image

DPI = 200


def convert_file(file_path: str, output_folder: str) -> list:
    """
    Converts a single file (PDF or image) into one or more normalized
    grayscale PNG images saved into output_folder.

    Returns a list of saved image paths (as strings).
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extension == ".pdf":
        return _convert_pdf(path, output_dir)
    elif extension in {".jpg", ".jpeg", ".png"}:
        return [_convert_image(path, output_dir)]
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def _convert_pdf(path: Path, output_dir: Path) -> list:
    saved_paths = []
    doc = fitz.open(path)

    # Matrix scales the render resolution. 72 is PDF's default DPI,
    # so (DPI / 72) gives us the zoom factor needed to reach our target DPI.
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_number in range(len(doc)):
        page = doc[page_number]
        # csGRAY renders directly in grayscale - no separate conversion step needed
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)

        output_name = f"{path.stem}_page_{page_number + 1}.png"
        output_path = output_dir / output_name
        pixmap.save(str(output_path))
        saved_paths.append(str(output_path))

    doc.close()
    return saved_paths


def _convert_image(path: Path, output_dir: Path) -> str:
    with Image.open(path) as img:
        grayscale_img = img.convert("L")  # "L" = 8-bit grayscale in Pillow

        output_name = f"{path.stem}_page_1.png"
        output_path = output_dir / output_name
        grayscale_img.save(output_path, format="PNG")

    return str(output_path)