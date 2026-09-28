# tests/test_input_validation.py

import fitz
from PIL import Image
from app.input.file_validator import validate_file


def test_valid_single_page_pdf(tmp_path):
    pdf_path = tmp_path / "single.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    result = validate_file(str(pdf_path))
    assert result["status"] == "valid"
    assert result["format"] == "pdf"
    assert result["page_count"] == 1


def test_valid_multi_page_pdf(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    result = validate_file(str(pdf_path))
    assert result["status"] == "valid"
    assert result["page_count"] == 3


def test_encrypted_pdf(tmp_path):
    pdf_path = tmp_path / "locked.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(
        pdf_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="ownerpass",
        user_pw="userpass",  # a real password required to open it
    )
    doc.close()

    result = validate_file(str(pdf_path))
    assert result["status"] == "invalid"
    assert "encrypted" in result["reason"].lower() or "password" in result["reason"].lower()


def test_valid_jpg(tmp_path):
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), color="white").save(img_path)

    result = validate_file(str(img_path))
    assert result["status"] == "valid"
    assert result["format"] == "jpg"
    assert result["page_count"] == 1


def test_valid_png(tmp_path):
    img_path = tmp_path / "photo.png"
    Image.new("RGB", (100, 100), color="white").save(img_path)

    result = validate_file(str(img_path))
    assert result["status"] == "valid"
    assert result["format"] == "png"


def test_unsupported_extension(tmp_path):
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("just some text")

    result = validate_file(str(bad_path))
    assert result["status"] == "invalid"
    assert "unsupported" in result["reason"].lower()


def test_corrupted_pdf(tmp_path):
    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"this is not a real pdf file")

    result = validate_file(str(fake_pdf))
    assert result["status"] == "invalid"


def test_corrupted_image(tmp_path):
    fake_img = tmp_path / "broken.jpg"
    fake_img.write_bytes(b"this is not a real image file")

    result = validate_file(str(fake_img))
    assert result["status"] == "invalid"


def test_nonexistent_file():
    result = validate_file("this/path/does/not/exist.pdf")
    assert result["status"] == "invalid"
    assert "does not exist" in result["reason"].lower()