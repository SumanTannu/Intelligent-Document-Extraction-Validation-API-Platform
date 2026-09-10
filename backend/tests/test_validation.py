from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfWriter

from app.services.document_validation_service import validate_document


def make_pdf(page_count: int) -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def make_image(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(output, format=image_format)
    return output.getvalue()


def test_valid_pdf_with_at_most_three_pages() -> None:
    result = validate_document("invoice.pdf", make_pdf(3))

    assert result == {
        "file_name": "invoice.pdf",
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 3,
            "status": "PASS",
        },
    }


def test_pdf_with_more_than_three_pages_is_rejected() -> None:
    result = validate_document("long-report.pdf", make_pdf(4))

    validation = result["file_validation"]
    assert validation["is_readable"] is True
    assert validation["page_count"] == 4
    assert validation["status"] == "FAIL"


def test_unsupported_file_type_is_rejected() -> None:
    result = validate_document("notes.txt", b"plain text")

    validation = result["file_validation"]
    assert validation["is_supported"] is False
    assert validation["is_readable"] is False
    assert validation["status"] == "FAIL"


def test_empty_file_is_rejected() -> None:
    result = validate_document("empty.pdf", b"")

    validation = result["file_validation"]
    assert validation["is_supported"] is True
    assert validation["is_readable"] is False
    assert validation["status"] == "FAIL"


def test_corrupted_pdf_is_rejected() -> None:
    result = validate_document("corrupted.pdf", b"%PDF-1.7\nnot a valid PDF")

    validation = result["file_validation"]
    assert validation["is_supported"] is True
    assert validation["is_readable"] is False
    assert validation["page_count"] is None
    assert validation["status"] == "FAIL"


@pytest.mark.parametrize(
    ("file_name", "image_format", "expected_type"),
    [
        ("scan.png", "PNG", "image/png"),
        ("scan.jpg", "JPEG", "image/jpeg"),
        ("scan.jpeg", "JPEG", "image/jpeg"),
    ],
)
def test_valid_png_and_jpeg_inputs(
    file_name: str,
    image_format: str,
    expected_type: str,
) -> None:
    result = validate_document(file_name, make_image(image_format))

    assert result["file_validation"] == {
        "file_type": expected_type,
        "is_supported": True,
        "is_readable": True,
        "page_count": None,
        "status": "PASS",
    }
