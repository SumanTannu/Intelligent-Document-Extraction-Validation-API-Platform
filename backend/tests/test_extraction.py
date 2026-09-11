from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.services.document_service import DocumentValidationError, process_document
from app.services.ocr_service import (
    TextExtractionError,
    _configure_tesseract_command,
    _correct_common_financial_text,
    _lost_substantive_rows,
    _normalize_financial_number_candidate,
    _reconcile_balance_sheet,
    _reconstruct_financial_tables,
    _restore_lost_substantive_rows,
    extract_text,
    normalize_financial_numbers,
)


OCR_TEXT = "SCANNED INVOICE TOTAL 1250"
NATIVE_TEXT = "Native invoice text with an amount of 1250 dollars"
DATASET_CASH_FLOW_2017 = (
    Path(__file__).resolve().parents[2]
    / "New Dataset"
    / "Cash Flows"
    / "Consolidated Cash Flow Statement 2017.pdf"
)


def test_tesseract_is_discovered_from_standard_windows_install(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")

    assert _configure_tesseract_command() == str(executable)
    assert pytesseract.pytesseract.tesseract_cmd == str(executable)


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
    text = (
        "As at March 31, 2020\n"
        "Years 2023 2024\n"
        "Values 125 300\n"
        "Reference 1,000 and 200"
    )

    assert normalize_financial_numbers(text) == text


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1 758,103,766", "1,758,103,766"),
        ("11,462,071 336", "11,462,071,336"),
        ("1,577 327,790", "1,577,327,790"),
        ("2,894,458, 722", "2,894,458,722"),
    ],
)
def test_numeric_cell_normalization_repairs_detached_groups(
    raw_value: str,
    expected: str,
) -> None:
    assert _normalize_financial_number_candidate(raw_value) == expected


def test_balance_sheet_reconciliation_selects_matching_ocr_alternative() -> None:
    lines = [
        {"text": "Consolidated Balance Sheet"},
        {"text": "CAPITAL AND LIABILITIES"},
        {"text": "Cash and balances with Reserve Bank of India 1,000 1,000"},
        {"text": "Balances with banks and money at call and short notice 2,000 2,000"},
        {"text": "Investments 3,000 3,000"},
        {"text": "Advances 4,000 4,000"},
        {"text": "Fixed assets 5,000 5,600"},
        {"text": "Other assets 6,000 6,000"},
        {"text": "Total 21,000 21,000"},
    ]

    corrections = _reconcile_balance_sheet(
        {"lines": lines},
        {6: {1: "5,000"}},
    )

    assert corrections == 1
    assert lines[6]["text"] == "Fixed assets 5,000 5,000"


def _word(text: str, x: int, y: int, width: int = 80) -> dict:
    return {
        "text": text,
        "confidence": 90.0,
        "bounding_box": {"x": x, "y": y, "width": width, "height": 30},
        "block_number": 1,
        "paragraph_number": 1,
        "line_number": 1,
    }


def _schedule_referenced_table_row() -> dict:
    raw_text = (
        "Cash equivalents on transfer [Refer Schedule 12(1)]"
        "        295,617        -"
    )
    return {
        "raw_text": raw_text,
        "text": raw_text,
        "bounding_box": {"x": 50, "y": 500, "width": 900, "height": 35},
        "words": [
            _word("Cash", 50, 500),
            _word("equivalents", 140, 500, 130),
            _word("on", 280, 500, 35),
            _word("transfer", 325, 500, 90),
            _word("[Refer", 425, 500, 70),
            _word("Schedule", 505, 500, 90),
            _word("12(1)]", 605, 500, 65),
            _word("295,617", 710, 500, 90),
            _word("-", 900, 500, 15),
        ],
    }


def test_header_recovery_does_not_replace_schedule_referenced_row() -> None:
    financial_row = _schedule_referenced_table_row()
    primary_header = {
        "raw_text": "Schedule 31-Mar-17 31-Mar-16",
        "text": "Schedule 31-Mar-17 31-Mar-16",
        "bounding_box": {"x": 500, "y": 100, "width": 450, "height": 35},
        "words": [],
    }
    primary_page = {
        "width": 1000,
        "height": 1000,
        "lines": [financial_row, primary_header],
    }
    secondary_page = {
        "width": 1000,
        "height": 1000,
        "text": "Schedule As at 31-Mar-17 As at 31-Mar-16",
        "lines": [
            {
                "raw_text": "Schedule As at 31-Mar-17 As at 31-Mar-16",
                "text": "Schedule As at 31-Mar-17 As at 31-Mar-16",
                "bounding_box": {
                    "x": 500,
                    "y": 100,
                    "width": 450,
                    "height": 35,
                },
                "words": [],
            }
        ],
    }

    _correct_common_financial_text(primary_page, secondary_page)

    assert financial_row["text"] == financial_row["raw_text"]
    assert primary_header["text"] == (
        "Schedule        As at 31-Mar-17        As at 31-Mar-16"
    )


def test_substantive_raw_row_is_restored_if_normalization_loses_it() -> None:
    original = _schedule_referenced_table_row()
    overwritten = _schedule_referenced_table_row()
    overwritten["text"] = "Schedule As at 31-Mar-17 As at 31-Mar-16"
    page = {"width": 1000, "height": 1000, "lines": [overwritten]}

    restored = _restore_lost_substantive_rows(page, [original])

    assert restored == 1
    assert page["lines"][0]["text"] == original["raw_text"]
    assert _lost_substantive_rows(page, [original]) == []


def test_wrapped_financial_label_is_joined_to_coordinate_aligned_values() -> None:
    def word(text: str, x: int, y: int) -> dict:
        return {
            "text": text,
            "confidence": 95.0,
            "bounding_box": {"x": x, "y": y, "width": max(20, len(text) * 8), "height": 25},
        }

    lines = [
        {
            "raw_text": "Increase in borrowings (excluding subordinate debt,",
            "text": "Increase in borrowings (excluding subordinate debt,",
            "bounding_box": {"x": 50, "y": 100, "width": 390, "height": 25},
            "words": [word("Increase", 50, 100), word("borrowings", 150, 100)],
        },
        {
            "raw_text": "(33,898,658) 402,081,134",
            "text": "(33,898,658)        402,081,134",
            "bounding_box": {"x": 650, "y": 125, "width": 300, "height": 25},
            "words": [word("(33,898,658)", 650, 125), word("402,081,134", 830, 125)],
        },
        {
            "raw_text": "perpetual debt and upper tier II instruments)",
            "text": "perpetual debt and upper tier II instruments)",
            "bounding_box": {"x": 50, "y": 150, "width": 390, "height": 25},
            "words": [word("perpetual", 50, 150), word("instruments)", 180, 150)],
        },
    ]
    page = {"width": 1000, "height": 1000, "lines": lines}

    reconstructed, regions = _reconstruct_financial_tables(page)

    assert reconstructed == 1
    assert len(page["lines"]) == 1
    assert page["lines"][0]["table_row"]["label"] == (
        "Increase in borrowings (excluding subordinate debt, "
        "perpetual debt and upper tier II instruments)"
    )
    assert page["lines"][0]["table_row"]["values"] == ["(33,898,658)", "402,081,134"]
    assert regions[0]["numeric_columns"]
    assert len(page["lines"][0]["source_lines"]) == 3


@pytest.mark.skipif(
    not DATASET_CASH_FLOW_2017.is_file(),
    reason="Local OCR dataset is not available.",
)
def test_cash_flow_2017_retains_amalgamation_row() -> None:
    result = extract_text(
        DATASET_CASH_FLOW_2017.name,
        DATASET_CASH_FLOW_2017.read_bytes(),
    )

    expected_row = (
        "Cash and cash equivalents on amalgamation [Refer Schedule 18(1)]"
        "                295,617                -"
    )
    assert expected_row in result["raw_text"]
    assert expected_row in result["extracted_text"]
    assert "₹ in '000" in result["extracted_text"]
    assert "Provision for diminution in value of Investments" in result["extracted_text"]
    assert "P. B. Pardiwalla" in result["extracted_text"]
    assert "LF HDFC BANK" not in result["extracted_text"]
    assert "1} HDFC BANK" not in result["extracted_text"]
    assert not any(
        line["text"].replace(" ", "") == "."
        for page in result["pages"]
        for line in page["lines"]
    )
    assert (
        "Increase / (decrease) in borrowings (excluding subordinate debt, "
        "perpetual debt and upper tier II instruments)"
    ) in result["extracted_text"]
    assert all(page["source_lines"] for page in result["pages"])
    assert all(page["table_regions"] for page in result["pages"])
    matching_rows = [
        row
        for page in result["pages"]
        for region in page["table_regions"]
        for row in region["rows"]
        if "amalgamation" in row["label"].casefold()
    ]
    assert len(matching_rows) == 1
    assert matching_rows[0]["label"] == (
        "Cash and cash equivalents on amalgamation [Refer Schedule 18(1)]"
    )
    assert matching_rows[0]["values"] == ["295,617", "-"]
    assert all(
        page["enhancement"]["lost_substantive_rows"] == 0
        for page in result["pages"]
    )


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
