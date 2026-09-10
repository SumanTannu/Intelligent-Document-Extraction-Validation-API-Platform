from io import BytesIO

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.services.document_service import DocumentValidationError, process_document
from app.services.ocr_service import (
    TextExtractionError,
    extract_text,
    normalize_financial_numbers,
)


OCR_TEXT = "SCANNED INVOICE TOTAL 1250"
NATIVE_TEXT = "Native invoice text with an amount of 1250 dollars"


def make_text_image(image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (1400, 300), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=52)
    draw.text((50, 100), OCR_TEXT, fill="black", font=font)

    output = BytesIO()
    image.save(output, format=image_format, quality=95)
    return output.getvalue()


def make_multicolumn_image() -> bytes:
    image = Image.new("RGB", (1600, 500), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    draw.text((60, 80), "LEFT DIRECTOR", fill="black", font=font)
    draw.text((900, 80), "RIGHT DIRECTOR", fill="black", font=font)
    draw.text((60, 260), "ALPHA NAME", fill="black", font=font)
    draw.text((900, 260), "BETA NAME", fill="black", font=font)

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_native_text_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), NATIVE_TEXT, fontsize=14)
    content = document.tobytes()
    document.close()
    return content


def make_scanned_pdf() -> bytes:
    image_content = make_text_image()
    document = pymupdf.open()
    page = document.new_page(width=700, height=150)
    page.insert_image(page.rect, stream=image_content)
    content = document.tobytes()
    document.close()
    return content


def make_blank_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (100, 100), color="white").save(output, format="PNG")
    return output.getvalue()


def test_native_text_pdf_extraction() -> None:
    result = extract_text("native.pdf", make_native_text_pdf())

    assert NATIVE_TEXT in result["extracted_text"]
    assert result["extraction_method"] == "native_text"
    assert result["page_count"] == 1
    assert result["status"] == "PASS"


def test_scanned_pdf_falls_back_to_ocr() -> None:
    result = extract_text("scanned.pdf", make_scanned_pdf())

    assert result["extracted_text"]
    assert "SCANNED INVOICE" in result["extracted_text"].upper()
    assert result["extraction_method"] == "ocr"
    assert result["page_count"] == 1
    assert result["ocr_details"]["render_dpi"] == 350


@pytest.mark.parametrize(
    ("file_name", "image_format"),
    [
        ("scan.png", "PNG"),
        ("scan.jpg", "JPEG"),
        ("scan.jpeg", "JPEG"),
    ],
)
def test_image_ocr(file_name: str, image_format: str) -> None:
    result = extract_text(file_name, make_text_image(image_format))

    assert "SCANNED INVOICE" in result["extracted_text"].upper()
    assert result["extraction_method"] == "ocr"
    assert result["page_count"] == 1


def test_grouped_financial_number_spacing_is_normalized() -> None:
    text = (
        "Current 8,923,441 ,607\n"
        "Prior 11,031 ,861,695\n"
        "Deposit 5,458, 732,889\n"
        "Rate 1,234 .50"
    )

    assert normalize_financial_numbers(text) == (
        "Current 8,923,441,607\n"
        "Prior 11,031,861,695\n"
        "Deposit 5,458,732,889\n"
        "Rate 1,234.50"
    )


def test_unrelated_numeric_spaces_are_not_removed() -> None:
    text = "Years 2023 2024\nValues 125 300\nReference 1,000 and 200"

    assert normalize_financial_numbers(text) == text


def test_multicolumn_ocr_retains_layout_and_positions() -> None:
    result = extract_text("directors.png", make_multicolumn_image())

    page = result["pages"][0]
    lines = page["lines"]
    assert "LEFT DIRECTOR" in result["extracted_text"].upper()
    assert "RIGHT DIRECTOR" in result["extracted_text"].upper()
    assert result["extracted_text"].upper().index("LEFT DIRECTOR") < (
        result["extracted_text"].upper().index("ALPHA NAME")
    )
    assert all(line["bounding_box"] for line in lines)
    assert all(word["bounding_box"] for line in lines for word in line["words"])
    assert all(
        isinstance(word["confidence"], float)
        for line in lines
        for word in line["words"]
    )
    assert any("  " in line["raw_text"] for line in lines)


def test_extraction_failure_is_controlled() -> None:
    with pytest.raises(
        TextExtractionError,
        match="No readable text could be extracted",
    ):
        extract_text("blank.png", make_blank_png())


def test_extraction_failure_returns_no_stack_trace() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("blank.png", make_blank_png(), "image/png")},
        data={"document_type": "invoice"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["text_extraction"]["status"] == "FAIL"
    assert "traceback" not in response.text.lower()
    assert "TextExtractionError" not in response.text


def test_native_text_is_preferred_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_ocr_is_called(_image: Image.Image, _page_number: int) -> None:
        raise AssertionError("OCR must not run for a readable native-text PDF.")

    monkeypatch.setattr(
        "app.services.ocr_service._ocr_image",
        fail_if_ocr_is_called,
    )

    result = extract_text("native.pdf", make_native_text_pdf())

    assert result["extraction_method"] == "native_text"
    assert NATIVE_TEXT in result["extracted_text"]


def test_document_service_validates_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_extraction_is_called(_file_name: str, _content: bytes) -> None:
        raise AssertionError("Extraction must not run after failed validation.")

    monkeypatch.setattr(
        "app.services.document_service.extract_text",
        fail_if_extraction_is_called,
    )

    with pytest.raises(DocumentValidationError):
        process_document("unsupported.txt", b"text")
