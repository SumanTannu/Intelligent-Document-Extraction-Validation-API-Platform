import os
import re
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pytesseract import Output, TesseractError, TesseractNotFoundError


DEFAULT_OCR_RENDER_DPI = 350
MIN_OCR_RENDER_DPI = 200
MAX_OCR_RENDER_DPI = 450
MIN_OCR_LONG_EDGE = 1800
NATIVE_TEXT_MIN_ALPHANUMERIC_CHARACTERS = 20
DEFAULT_TESSERACT_OEM = 3
DEFAULT_TESSERACT_PSM = 3
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_BROKEN_GROUPED_NUMBER = re.compile(
    r"(?<![\d,])(?P<head>\d{1,3}(?:,\d{3})+)\s+,\s*(?P<tail>\d{3})(?!\d)"
)
_BROKEN_GROUPED_NUMBER_AFTER_COMMA = re.compile(
    r"(?<![\d,])(?P<head>\d{1,3}(?:,\d{3})+),\s+"
    r"(?P<tail>\d{3}(?:,\d{3})+)(?!\d)"
)
_BROKEN_GROUPED_DECIMAL = re.compile(
    r"(?<![\d,])(?P<head>\d{1,3}(?:,\d{3})+)\s+\.\s*(?P<fraction>\d{2})(?!\d)"
)


class TextExtractionError(Exception):
    """A controlled failure raised when readable text cannot be extracted."""


def extract_text(file_name: str, content: bytes) -> dict[str, Any]:
    """Extract text and layout metadata from a validated PDF or image."""
    suffix = Path(file_name).suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(content)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return _extract_from_image(content)

    raise TextExtractionError("Text extraction is not supported for this file type.")


def normalize_financial_numbers(text: str) -> str:
    """Repair only strongly evidenced breaks in already-grouped financial numbers."""
    normalized = text
    while True:
        updated = _BROKEN_GROUPED_NUMBER.sub(
            lambda match: f"{match.group('head')},{match.group('tail')}",
            normalized,
        )
        updated = _BROKEN_GROUPED_NUMBER_AFTER_COMMA.sub(
            lambda match: f"{match.group('head')},{match.group('tail')}",
            updated,
        )
        if updated == normalized:
            break
        normalized = updated

    return _BROKEN_GROUPED_DECIMAL.sub(
        lambda match: f"{match.group('head')}.{match.group('fraction')}",
        normalized,
    )


def _extract_from_pdf(content: bytes) -> dict[str, Any]:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            page_count = document.page_count
            native_pages = [
                {
                    "page_number": page_number,
                    "text": page.get_text("text").strip(),
                    "status": "PASS",
                }
                for page_number, page in enumerate(document, start=1)
            ]
            native_text = _join_page_text(
                [page["text"] for page in native_pages]
            )

            if _has_sufficient_native_text(native_text):
                return _success(
                    native_text,
                    "native_text",
                    page_count,
                    pages=native_pages,
                )

            render_dpi = _integer_setting(
                "OCR_RENDER_DPI",
                DEFAULT_OCR_RENDER_DPI,
                MIN_OCR_RENDER_DPI,
                MAX_OCR_RENDER_DPI,
            )
            ocr_pages = [
                _ocr_pdf_page(page, page_number, render_dpi)
                for page_number, page in enumerate(document, start=1)
            ]
    except TextExtractionError:
        raise
    except Exception as exc:
        raise TextExtractionError(
            "Text extraction failed because the PDF could not be processed."
        ) from exc

    raw_text = _join_page_text([page["raw_text"] for page in ocr_pages])
    extracted_text = _join_page_text([page["text"] for page in ocr_pages])
    if not _has_readable_text(extracted_text):
        raise TextExtractionError("No readable text could be extracted from the PDF.")

    return _success(
        extracted_text,
        "ocr",
        page_count,
        raw_text=raw_text,
        pages=ocr_pages,
        ocr_details=_ocr_details(render_dpi),
    )


def _extract_from_image(content: bytes) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            ocr_page = _ocr_image(image, page_number=1)
    except TextExtractionError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise TextExtractionError(
            "Text extraction failed because the image could not be processed."
        ) from exc

    extracted_text = ocr_page["text"]
    if not _has_readable_text(extracted_text):
        raise TextExtractionError("No readable text could be extracted from the image.")

    return _success(
        extracted_text,
        "ocr",
        1,
        raw_text=ocr_page["raw_text"],
        pages=[ocr_page],
        ocr_details=_ocr_details(render_dpi=None),
    )


def _ocr_pdf_page(
    page: pymupdf.Page,
    page_number: int,
    render_dpi: int,
) -> dict[str, Any]:
    scale = render_dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
        image.load()
        return _ocr_image(image, page_number)


def _ocr_image(image: Image.Image, page_number: int) -> dict[str, Any]:
    prepared_image = _preprocess_image(image)
    data = _run_tesseract(prepared_image)
    return _build_ocr_page(
        data,
        page_number=page_number,
        width=prepared_image.width,
        height=prepared_image.height,
    )


def _preprocess_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    longest_edge = max(grayscale.size)
    if longest_edge < MIN_OCR_LONG_EDGE:
        scale = min(2.0, MIN_OCR_LONG_EDGE / longest_edge)
        resized_size = (
            max(1, round(grayscale.width * scale)),
            max(1, round(grayscale.height * scale)),
        )
        grayscale = grayscale.resize(resized_size, Image.Resampling.LANCZOS)

    enhanced = ImageOps.autocontrast(grayscale)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)
    return enhanced.filter(
        ImageFilter.UnsharpMask(radius=1, percent=125, threshold=3)
    )


def _run_tesseract(image: Image.Image) -> dict[str, list[Any]]:
    configured_command = os.getenv("TESSERACT_CMD")
    if configured_command:
        pytesseract.pytesseract.tesseract_cmd = configured_command

    try:
        return pytesseract.image_to_data(
            image,
            lang=os.getenv("TESSERACT_LANG", "eng"),
            config=_tesseract_config(),
            output_type=Output.DICT,
        )
    except TesseractNotFoundError as exc:
        raise TextExtractionError(
            "The Tesseract OCR engine is not installed or configured."
        ) from exc
    except TesseractError as exc:
        raise TextExtractionError("The Tesseract OCR engine could not extract text.") from exc


def _tesseract_config() -> str:
    oem = _integer_setting("TESSERACT_OEM", DEFAULT_TESSERACT_OEM, 0, 3)
    psm = _integer_setting("TESSERACT_PSM", DEFAULT_TESSERACT_PSM, 0, 13)
    extra_config = os.getenv("TESSERACT_EXTRA_CONFIG", "").strip()
    parts = [f"--oem {oem}", f"--psm {psm}", "-c preserve_interword_spaces=1"]
    if extra_config:
        parts.append(extra_config)
    return " ".join(parts)


def _build_ocr_page(
    data: dict[str, list[Any]],
    *,
    page_number: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    words = _ocr_words(data)
    lines = _visual_lines(words)
    raw_text = "\n".join(line["raw_text"] for line in lines)

    return {
        "page_number": page_number,
        "width": width,
        "height": height,
        "raw_text": raw_text,
        "text": normalize_financial_numbers(raw_text),
        "status": "PASS" if _has_readable_text(raw_text) else "FAIL",
        "lines": lines,
    }


def _ocr_words(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue

        words.append(
            {
                "text": text,
                "confidence": float(data["conf"][index]),
                "bounding_box": {
                    "x": int(data["left"][index]),
                    "y": int(data["top"][index]),
                    "width": int(data["width"][index]),
                    "height": int(data["height"][index]),
                },
                "block_number": int(data["block_num"][index]),
                "paragraph_number": int(data["par_num"][index]),
                "line_number": int(data["line_num"][index]),
            }
        )
    return _deduplicate_overlapping_words(words)


def _deduplicate_overlapping_words(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            word["text"].casefold() == existing["text"].casefold()
            and _overlap_coverage(word["bounding_box"], existing["bounding_box"])
            >= 0.75
            for existing in retained
        )
        if not duplicate:
            retained.append(word)
    return retained


def _overlap_coverage(first: dict[str, int], second: dict[str, int]) -> float:
    intersection_width = max(
        0,
        min(first["x"] + first["width"], second["x"] + second["width"])
        - max(first["x"], second["x"]),
    )
    intersection_height = max(
        0,
        min(first["y"] + first["height"], second["y"] + second["height"])
        - max(first["y"], second["y"]),
    )
    intersection_area = intersection_width * intersection_height
    smaller_area = min(
        first["width"] * first["height"],
        second["width"] * second["height"],
    )
    return intersection_area / smaller_area if smaller_area else 0.0


def _visual_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for word in sorted(words, key=_word_position):
        matching_row = _matching_row(rows, word)
        if matching_row is None:
            box = word["bounding_box"]
            matching_row = {
                "words": [],
                "top": box["y"],
                "bottom": box["y"] + box["height"],
            }
            rows.append(matching_row)

        matching_row["words"].append(word)
        box = word["bounding_box"]
        matching_row["top"] = min(matching_row["top"], box["y"])
        matching_row["bottom"] = max(
            matching_row["bottom"], box["y"] + box["height"]
        )

    lines = [_line_from_row(row) for row in rows]
    return sorted(
        lines,
        key=lambda line: (
            line["bounding_box"]["y"],
            line["bounding_box"]["x"],
        ),
    )


def _matching_row(
    rows: list[dict[str, Any]],
    word: dict[str, Any],
) -> dict[str, Any] | None:
    box = word["bounding_box"]
    word_center = box["y"] + box["height"] / 2
    candidates = []
    for row in rows:
        row_center = (row["top"] + row["bottom"]) / 2
        row_height = row["bottom"] - row["top"]
        tolerance = max(6.0, min(box["height"], row_height) * 0.55)
        difference = abs(word_center - row_center)
        if difference <= tolerance:
            candidates.append((difference, row))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _line_from_row(row: dict[str, Any]) -> dict[str, Any]:
    words = sorted(row["words"], key=lambda word: word["bounding_box"]["x"])
    character_widths = [
        word["bounding_box"]["width"] / max(len(word["text"]), 1)
        for word in words
        if word["bounding_box"]["width"] > 0
    ]
    typical_character_width = median(character_widths) if character_widths else 1

    parts: list[str] = []
    previous_right: int | None = None
    for word in words:
        box = word["bounding_box"]
        if previous_right is not None:
            gap = box["x"] - previous_right
            spaces = max(1, min(16, round(gap / typical_character_width)))
            parts.append(" " * spaces)
        parts.append(word["text"])
        previous_right = box["x"] + box["width"]

    left = min(word["bounding_box"]["x"] for word in words)
    right = max(
        word["bounding_box"]["x"] + word["bounding_box"]["width"]
        for word in words
    )
    raw_text = "".join(parts)
    return {
        "raw_text": raw_text,
        "text": normalize_financial_numbers(raw_text),
        "bounding_box": {
            "x": left,
            "y": row["top"],
            "width": right - left,
            "height": row["bottom"] - row["top"],
        },
        "words": words,
    }


def _word_position(word: dict[str, Any]) -> tuple[float, int]:
    box = word["bounding_box"]
    return (box["y"] + box["height"] / 2, box["x"])


def _join_page_text(pages: list[str]) -> str:
    return "\n\n".join(text.strip() for text in pages if text.strip())


def _has_sufficient_native_text(text: str) -> bool:
    alphanumeric_count = sum(character.isalnum() for character in text)
    word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
    return (
        alphanumeric_count >= NATIVE_TEXT_MIN_ALPHANUMERIC_CHARACTERS
        and word_count >= 3
    )


def _has_readable_text(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _success(
    extracted_text: str,
    extraction_method: str,
    page_count: int,
    *,
    raw_text: str | None = None,
    pages: list[dict[str, Any]] | None = None,
    ocr_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "extracted_text": extracted_text.strip(),
        "extraction_method": extraction_method,
        "page_count": page_count,
        "status": "PASS",
    }
    if raw_text is not None:
        result["raw_text"] = raw_text.strip()
    if pages is not None:
        result["pages"] = pages
    if ocr_details is not None:
        result["ocr_details"] = ocr_details
    return result


def _ocr_details(render_dpi: int | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "language": os.getenv("TESSERACT_LANG", "eng"),
        "engine_mode": _integer_setting(
            "TESSERACT_OEM", DEFAULT_TESSERACT_OEM, 0, 3
        ),
        "page_segmentation_mode": _integer_setting(
            "TESSERACT_PSM", DEFAULT_TESSERACT_PSM, 0, 13
        ),
        "preprocessing": [
            "grayscale",
            "conditional_upscale",
            "autocontrast",
            "mild_contrast_enhancement",
            "unsharp_mask",
        ],
    }
    if render_dpi is not None:
        details["render_dpi"] = render_dpi
    return details


def _integer_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))
