import os
import re
from difflib import SequenceMatcher
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from pytesseract import Output, TesseractError, TesseractNotFoundError


DEFAULT_OCR_RENDER_DPI = 350
DEFAULT_SECONDARY_OCR_RENDER_DPI = 450
MIN_OCR_RENDER_DPI = 200
MAX_OCR_RENDER_DPI = 450
MIN_OCR_LONG_EDGE = 1800
NATIVE_TEXT_MIN_ALPHANUMERIC_CHARACTERS = 20
DEFAULT_TESSERACT_OEM = 3
DEFAULT_TESSERACT_PSM = 3
DEFAULT_SECONDARY_TESSERACT_PSM = 4
DEFAULT_SPARSE_TESSERACT_PSM = 11
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
_SPACE_AFTER_SEPARATOR = re.compile(r"(?<=\d),\s+(?=\d)")
_MISSING_FINAL_GROUP_SEPARATOR = re.compile(
    r"(?<![\d,])(?P<head>\d{1,3}(?:,\d{3})+)\s+(?P<tail>\d{3})(?![\d,])"
)
_MISSING_GROUP_SEPARATOR_BEFORE_GROUPED_TAIL = re.compile(
    r"(?<![\d,])(?P<head>\d{1,3}(?:,\d{3})*)\s+"
    r"(?P<tail>\d{3}(?:,\d{3})+)(?!\d)"
)
_DETACHED_LEADING_GROUP = re.compile(
    r"(?<!\d)(?P<head>\d{1,3})\s+(?P<tail>\d{3}(?:,\d{3})+)(?!\d)"
)
_STRICT_FINANCIAL_NUMBER = re.compile(
    r"^-?(?:\d{1,3}(?:,\d{3})+|\d{3,})(?:\.\d+)?$"
)
_GROUPED_FINANCIAL_NUMBER = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
)
_NUMERIC_OCR_TOKEN = re.compile(r"^[\d,\.\-()?]+$")
_SCHEDULE_VALUE = re.compile(r"^(?:\d{1,2}[A-Z]?|\d{1,2}&\d{1,2})$")
_SHORT_TABLE_DATE = re.compile(r"\b\d{2}-[A-Za-z]{3}-\d{2}\b")
_CURRENCY_THOUSANDS_ARTIFACT = re.compile(
    r"^\s*(?:Z|₹)\s*in\s+['‘’�]?[O0]{3}\s*$",
    flags=re.IGNORECASE,
)

_BALANCE_SHEET_LIABILITY_LABELS = (
    "capital",
    "reserves and surplus",
    "minority interest",
    "deposits",
    "borrowings",
    "other liabilities and provisions",
)
_BALANCE_SHEET_ASSET_LABELS = (
    "cash and balances with reserve bank of india",
    "balances with banks and money at call and short notice",
    "investments",
    "advances",
    "fixed assets",
    "other assets",
)
_FINANCIAL_SECTION_HEADINGS = {
    "ASSETS",
    "CAPITAL AND LIABILITIES",
    "CASH FLOWS",
    "LIABILITIES",
}


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


def _normalize_financial_number_candidate(text: str) -> str:
    normalized = normalize_financial_numbers(text.strip())
    normalized = _SPACE_AFTER_SEPARATOR.sub(",", normalized)
    while True:
        updated = _MISSING_FINAL_GROUP_SEPARATOR.sub(
            lambda match: f"{match.group('head')},{match.group('tail')}",
            normalized,
        )
        updated = _MISSING_GROUP_SEPARATOR_BEFORE_GROUPED_TAIL.sub(
            lambda match: f"{match.group('head')},{match.group('tail')}",
            updated,
        )
        updated = _DETACHED_LEADING_GROUP.sub(
            lambda match: f"{match.group('head')},{match.group('tail')}",
            updated,
        )
        if updated == normalized:
            return updated
        normalized = updated


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
            ocr_page = _ocr_image_with_multiple_passes(image, page_number=1)
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
    primary_image = _render_pdf_page(page, render_dpi)
    primary_page = _ocr_image(primary_image, page_number)
    if not _boolean_setting("OCR_MULTI_PASS", True):
        return primary_page

    secondary_dpi = _integer_setting(
        "OCR_SECONDARY_RENDER_DPI",
        DEFAULT_SECONDARY_OCR_RENDER_DPI,
        MIN_OCR_RENDER_DPI,
        MAX_OCR_RENDER_DPI,
    )
    secondary_psm = _integer_setting(
        "OCR_SECONDARY_PSM",
        DEFAULT_SECONDARY_TESSERACT_PSM,
        0,
        13,
    )
    secondary_image = (
        primary_image
        if secondary_dpi == render_dpi
        else _render_pdf_page(page, secondary_dpi)
    )
    secondary_page = _ocr_image(
        secondary_image,
        page_number,
        psm_override=secondary_psm,
    )
    sparse_page = _ocr_image(
        primary_image,
        page_number,
        psm_override=DEFAULT_SPARSE_TESSERACT_PSM,
    )
    return _enhance_ocr_page(
        primary_page,
        secondary_page,
        sparse_page,
        secondary_image,
    )


def _render_pdf_page(page: pymupdf.Page, render_dpi: int) -> Image.Image:
    scale = render_dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
        image.load()
        return image.copy()


def _ocr_image(
    image: Image.Image,
    page_number: int,
    *,
    psm_override: int | None = None,
) -> dict[str, Any]:
    prepared_image = _preprocess_image(image)
    return _ocr_prepared_image(
        prepared_image,
        page_number,
        psm_override=psm_override,
    )


def _ocr_prepared_image(
    prepared_image: Image.Image,
    page_number: int,
    *,
    psm_override: int | None = None,
) -> dict[str, Any]:
    data = _run_tesseract(prepared_image, psm_override=psm_override)
    return _build_ocr_page(
        data,
        page_number=page_number,
        width=prepared_image.width,
        height=prepared_image.height,
    )


def _ocr_image_with_multiple_passes(
    image: Image.Image,
    page_number: int,
) -> dict[str, Any]:
    prepared_image = _preprocess_image(image)
    primary_page = _ocr_prepared_image(prepared_image, page_number)
    if not _boolean_setting("OCR_MULTI_PASS", True):
        return primary_page

    secondary_page = _ocr_prepared_image(
        prepared_image,
        page_number,
        psm_override=_integer_setting(
            "OCR_SECONDARY_PSM",
            DEFAULT_SECONDARY_TESSERACT_PSM,
            0,
            13,
        ),
    )
    sparse_page = _ocr_prepared_image(
        prepared_image,
        page_number,
        psm_override=DEFAULT_SPARSE_TESSERACT_PSM,
    )
    return _enhance_ocr_page(
        primary_page,
        secondary_page,
        sparse_page,
        prepared_image,
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


def _run_tesseract(
    image: Image.Image,
    *,
    psm_override: int | None = None,
    extra_config_override: str | None = None,
) -> dict[str, list[Any]]:
    configured_command = os.getenv("TESSERACT_CMD")
    if configured_command:
        pytesseract.pytesseract.tesseract_cmd = configured_command

    try:
        return pytesseract.image_to_data(
            image,
            lang=os.getenv("TESSERACT_LANG", "eng"),
            config=_tesseract_config(
                psm_override=psm_override,
                extra_config_override=extra_config_override,
            ),
            output_type=Output.DICT,
        )
    except TesseractNotFoundError as exc:
        raise TextExtractionError(
            "The Tesseract OCR engine is not installed or configured."
        ) from exc
    except TesseractError as exc:
        raise TextExtractionError("The Tesseract OCR engine could not extract text.") from exc


def _tesseract_config(
    *,
    psm_override: int | None = None,
    extra_config_override: str | None = None,
) -> str:
    oem = _integer_setting("TESSERACT_OEM", DEFAULT_TESSERACT_OEM, 0, 3)
    psm = (
        psm_override
        if psm_override is not None
        else _integer_setting("TESSERACT_PSM", DEFAULT_TESSERACT_PSM, 0, 13)
    )
    extra_config = (
        extra_config_override
        if extra_config_override is not None
        else os.getenv("TESSERACT_EXTRA_CONFIG", "").strip()
    )
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


def _enhance_ocr_page(
    primary_page: dict[str, Any],
    secondary_page: dict[str, Any],
    sparse_page: dict[str, Any],
    secondary_image: Image.Image,
) -> dict[str, Any]:
    alternatives, numeric_replacements = _merge_financial_number_passes(
        primary_page,
        secondary_page,
    )
    text_replacements = _improve_nonfinancial_text(
        primary_page,
        secondary_page,
        sparse_page,
    )
    schedule_replacements = _recover_schedule_values(
        primary_page,
        secondary_image,
    )
    reconciled_values = _reconcile_balance_sheet(primary_page, alternatives)
    heading_recoveries = _recover_financial_section_headings(
        primary_page,
        sparse_page,
    )
    common_text_corrections = _correct_common_financial_text(
        primary_page,
        secondary_page,
    )
    removed_layout_noise = _remove_layout_noise(primary_page)
    primary_page["text"] = _join_page_text(
        [line["text"] for line in primary_page["lines"]]
    )
    primary_page["enhancement"] = {
        "multi_pass": True,
        "numeric_replacements": numeric_replacements,
        "schedule_replacements": schedule_replacements,
        "text_replacements": text_replacements,
        "recovered_headings": heading_recoveries,
        "reconciled_values": reconciled_values,
        "common_text_corrections": common_text_corrections,
        "removed_layout_noise": removed_layout_noise,
    }
    return primary_page


def _merge_financial_number_passes(
    primary_page: dict[str, Any],
    secondary_page: dict[str, Any],
) -> tuple[dict[int, dict[int, str]], int]:
    alternatives: dict[int, dict[int, str]] = {}
    replacement_count = 0
    primary_width = primary_page["width"]
    primary_height = primary_page["height"]
    secondary_width = secondary_page["width"]
    secondary_height = secondary_page["height"]

    for line_index, primary_line in enumerate(primary_page["lines"]):
        primary_groups = _financial_number_groups(primary_line, primary_width)
        if not primary_groups:
            continue

        amount_groups = primary_groups[-2:]
        primary_values = []
        for primary_group in amount_groups:
            primary_value = _normalize_financial_number_candidate(
                primary_group["text"]
            )
            primary_values.append(primary_value)
            if _is_strict_financial_number(primary_value):
                updated = _replace_flexible(
                    primary_line["text"],
                    primary_group["text"],
                    primary_value,
                )
                if updated != primary_line["text"]:
                    primary_line["text"] = updated
                    replacement_count += 1

        secondary_line = _nearest_line(
            primary_line,
            secondary_page["lines"],
            primary_height,
            secondary_height,
        )
        if secondary_line is None:
            continue
        secondary_groups = _financial_number_groups(
            secondary_line,
            secondary_width,
        )
        if not secondary_groups:
            continue

        for column, primary_group in enumerate(amount_groups):
            primary_value = primary_values[column]

            secondary_group = _nearest_number_group(
                primary_group,
                secondary_groups,
                primary_width,
                secondary_width,
            )
            if secondary_group is None:
                continue

            secondary_value = _normalize_financial_number_candidate(
                secondary_group["text"]
            )
            if not _is_strict_financial_number(secondary_value):
                continue

            alternatives.setdefault(line_index, {})[column] = secondary_value
            if _is_strict_financial_number(primary_value):
                continue
            if secondary_group["confidence"] < 35:
                continue

            updated = _replace_flexible(
                primary_line["text"],
                primary_group["text"],
                secondary_value,
            )
            if updated != primary_line["text"]:
                primary_line["text"] = updated
                replacement_count += 1

    return alternatives, replacement_count


def _financial_number_groups(
    line: dict[str, Any],
    page_width: int,
) -> list[dict[str, Any]]:
    candidates = []
    for word in sorted(line.get("words", []), key=lambda item: item["bounding_box"]["x"]):
        box = word["bounding_box"]
        center_x = box["x"] + box["width"] / 2
        text = word["text"].strip()
        if center_x / page_width < 0.65:
            continue
        if not any(character.isdigit() for character in text):
            continue
        if not _NUMERIC_OCR_TOKEN.fullmatch(text):
            continue
        candidates.append(word)

    groups: list[list[dict[str, Any]]] = []
    for word in candidates:
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]["bounding_box"]
        current = word["bounding_box"]
        gap = current["x"] - (previous["x"] + previous["width"])
        if gap <= page_width * 0.025:
            groups[-1].append(word)
        else:
            groups.append([word])

    return [_number_group(group) for group in groups]


def _number_group(words: list[dict[str, Any]]) -> dict[str, Any]:
    left = min(word["bounding_box"]["x"] for word in words)
    right = max(
        word["bounding_box"]["x"] + word["bounding_box"]["width"]
        for word in words
    )
    weights = [max(len(word["text"]), 1) for word in words]
    confidence = sum(
        word["confidence"] * weight for word, weight in zip(words, weights)
    ) / sum(weights)
    return {
        "text": " ".join(word["text"] for word in words),
        "x": left,
        "right": right,
        "confidence": confidence,
    }


def _nearest_line(
    primary_line: dict[str, Any],
    secondary_lines: list[dict[str, Any]],
    primary_height: int,
    secondary_height: int,
) -> dict[str, Any] | None:
    primary_box = primary_line["bounding_box"]
    primary_center = (
        primary_box["y"] + primary_box["height"] / 2
    ) / primary_height
    matches = []
    for line in secondary_lines:
        box = line["bounding_box"]
        center = (box["y"] + box["height"] / 2) / secondary_height
        difference = abs(primary_center - center)
        if difference <= 0.018:
            matches.append((difference, line))
    if not matches:
        return None
    return min(matches, key=lambda match: match[0])[1]


def _nearest_number_group(
    primary_group: dict[str, Any],
    secondary_groups: list[dict[str, Any]],
    primary_width: int,
    secondary_width: int,
) -> dict[str, Any] | None:
    primary_center = (
        primary_group["x"] + primary_group["right"]
    ) / (2 * primary_width)
    matches = []
    for group in secondary_groups:
        center = (group["x"] + group["right"]) / (2 * secondary_width)
        difference = abs(primary_center - center)
        if difference <= 0.06:
            matches.append((difference, group))
    if not matches:
        return None
    return min(matches, key=lambda match: match[0])[1]


def _improve_nonfinancial_text(
    primary_page: dict[str, Any],
    secondary_page: dict[str, Any],
    sparse_page: dict[str, Any],
) -> int:
    replacements = 0
    for primary_line in primary_page["lines"]:
        if len(_financial_number_groups(primary_line, primary_page["width"])) >= 2:
            continue

        primary_tokens = primary_line["text"].split()
        if not primary_tokens:
            continue
        primary_confidence = _line_confidence(primary_line)
        candidates = []
        for alternate_page in (secondary_page, sparse_page):
            alternate_line = _nearest_line(
                primary_line,
                alternate_page["lines"],
                primary_page["height"],
                alternate_page["height"],
            )
            if alternate_line is None:
                continue
            alternate_tokens = alternate_line["text"].split()
            similarity = SequenceMatcher(
                None,
                [token.casefold() for token in primary_tokens],
                [token.casefold() for token in alternate_tokens],
            ).ratio()
            if similarity < 0.7:
                continue
            candidates.append(
                (_line_confidence(alternate_line), alternate_line, similarity)
            )
        if not candidates:
            continue

        proposals: dict[tuple[str, str], dict[str, float]] = {}
        for alternate_confidence, alternate_line, _ in candidates:
            if alternate_confidence < primary_confidence - 1:
                continue
            alternate_tokens = alternate_line["text"].split()
            matcher = SequenceMatcher(
                None,
                [token.casefold() for token in primary_tokens],
                [token.casefold() for token in alternate_tokens],
            )
            for operation, first_start, first_end, second_start, second_end in (
                matcher.get_opcodes()
            ):
                if operation != "replace":
                    continue
                original = " ".join(primary_tokens[first_start:first_end])
                replacement = " ".join(alternate_tokens[second_start:second_end])
                if not original or not replacement:
                    continue
                proposal = proposals.setdefault(
                    (original, replacement),
                    {"votes": 0, "confidence": alternate_confidence},
                )
                proposal["votes"] += 1
                proposal["confidence"] = max(
                    proposal["confidence"],
                    alternate_confidence,
                )

        updated = primary_line["text"]
        for (original, replacement), evidence in proposals.items():
            is_supported_by_both_passes = evidence["votes"] >= 2
            is_high_confidence_extension = (
                replacement.casefold().startswith(original.casefold())
                and len(replacement) <= len(original) + 2
                and evidence["confidence"] >= primary_confidence + 2
            )
            if not is_supported_by_both_passes and not is_high_confidence_extension:
                continue
            changed = _replace_flexible(updated, original, replacement)
            if changed != updated:
                updated = changed
                replacements += 1
        primary_line["text"] = updated
    return replacements


def _line_confidence(line: dict[str, Any]) -> float:
    words = line.get("words", [])
    if not words:
        return 0.0
    weights = [max(len(word["text"]), 1) for word in words]
    return sum(
        word["confidence"] * weight for word, weight in zip(words, weights)
    ) / sum(weights)


def _recover_schedule_values(
    page: dict[str, Any],
    secondary_image: Image.Image,
) -> int:
    schedule_words = [
        word
        for line in page["lines"]
        for word in line.get("words", [])
        if word["text"].casefold() == "schedule"
    ]
    if not schedule_words:
        return 0

    prepared_image = ImageOps.grayscale(secondary_image)
    longest_edge = max(prepared_image.size)
    if longest_edge < MIN_OCR_LONG_EDGE:
        scale = min(2.0, MIN_OCR_LONG_EDGE / longest_edge)
        prepared_image = prepared_image.resize(
            (
                max(1, round(prepared_image.width * scale)),
                max(1, round(prepared_image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    prepared_image = ImageOps.autocontrast(prepared_image)
    prepared_image = ImageEnhance.Contrast(prepared_image).enhance(1.15)
    page_width = page["width"]
    page_height = page["height"]
    schedule_box = schedule_words[0]["bounding_box"]
    schedule_center = (schedule_box["x"] + schedule_box["width"] / 2) / page_width
    replacements = 0

    for line in page["lines"]:
        amount_groups = _financial_number_groups(line, page_width)
        is_notes_schedule = "policies and notes" in line["text"].casefold()
        if len(amount_groups) < 2 and not is_notes_schedule:
            continue

        existing_words = []
        for word in line.get("words", []):
            word_box = word["bounding_box"]
            center = (word_box["x"] + word_box["width"] / 2) / page_width
            if abs(center - schedule_center) <= 0.04:
                existing_words.append(word)
        if not existing_words:
            continue
        existing_words.sort(key=lambda word: word["bounding_box"]["x"])
        existing = "".join(word["text"] for word in existing_words)
        existing_confidence = sum(
            word["confidence"] * max(len(word["text"]), 1)
            for word in existing_words
        ) / sum(max(len(word["text"]), 1) for word in existing_words)
        if _SCHEDULE_VALUE.fullmatch(existing.upper()) and existing_confidence >= 80:
            continue

        box = line["bounding_box"]
        crop_box = (
            round(max(0, schedule_center - 0.04) * prepared_image.width),
            round(max(0, box["y"] / page_height - 0.002) * prepared_image.height),
            round(min(1, schedule_center + 0.04) * prepared_image.width),
            round(
                min(1, (box["y"] + box["height"]) / page_height + 0.002)
                * prepared_image.height
            ),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        data = _run_tesseract(
            prepared_image.crop(crop_box),
            psm_override=7,
            extra_config_override=(
                "-c tessedit_char_whitelist=0123456789A&"
            ),
        )
        recognized = [
            (str(text).strip(), float(confidence))
            for text, confidence in zip(data.get("text", []), data.get("conf", []))
            if str(text).strip()
        ]
        if not recognized:
            continue
        candidate = "".join(text for text, _ in recognized).upper()
        candidate_confidence = sum(
            confidence * max(len(text), 1) for text, confidence in recognized
        ) / sum(max(len(text), 1) for text, _ in recognized)
        if not _SCHEDULE_VALUE.fullmatch(candidate) or candidate_confidence < 65:
            continue

        if candidate == existing.upper():
            continue
        if candidate_confidence < existing_confidence + 10:
            continue

        updated = _replace_flexible(line["text"], existing, candidate)
        if updated != line["text"]:
            line["text"] = updated
            replacements += 1

    return replacements


def _recover_financial_section_headings(
    primary_page: dict[str, Any],
    sparse_page: dict[str, Any],
) -> int:
    existing = {line["text"].strip().upper() for line in primary_page["lines"]}
    recovered = []
    for line in sparse_page["lines"]:
        heading = line["text"].strip().upper()
        if heading in _FINANCIAL_SECTION_HEADINGS and heading not in existing:
            recovered.append(line)
            existing.add(heading)
    if recovered:
        primary_page["lines"].extend(recovered)
        primary_page["lines"].sort(
            key=lambda line: (
                line["bounding_box"]["y"],
                line["bounding_box"]["x"],
            )
        )
    return len(recovered)


def _correct_common_financial_text(
    primary_page: dict[str, Any],
    secondary_page: dict[str, Any],
) -> int:
    corrections = 0
    for line in primary_page["lines"]:
        updated = re.sub(
            r"\bon behalf af the Board\b",
            "on behalf of the Board",
            line["text"],
            flags=re.IGNORECASE,
        )
        updated = re.sub(r"^(Total\s+)_\s+", r"\1", updated)
        if _CURRENCY_THOUSANDS_ARTIFACT.fullmatch(updated):
            updated = "₹ in '000"
        if updated != line["text"]:
            line["text"] = updated
            corrections += 1

    secondary_dates = _SHORT_TABLE_DATE.findall(secondary_page["text"])
    if len(secondary_dates) >= 2:
        for line in primary_page["lines"]:
            if "schedule" not in line["text"].casefold():
                continue
            recovered_header = (
                f"Schedule        As at {secondary_dates[0]}"
                f"        As at {secondary_dates[1]}"
            )
            if line["text"] != recovered_header:
                line["text"] = recovered_header
                corrections += 1
            break
    return corrections


def _remove_layout_noise(page: dict[str, Any]) -> int:
    original_count = len(page["lines"])
    schedule_centers = [
        (line["bounding_box"]["y"] + line["bounding_box"]["height"] / 2)
        / page["height"]
        for line in page["lines"]
        if "schedule" in line["text"].casefold()
        and len(_SHORT_TABLE_DATE.findall(line["text"])) >= 2
    ]

    retained = []
    for line in page["lines"]:
        stripped = line["text"].strip()
        compact = stripped.replace(" ", "")
        if compact and all(character in "iIl|/;:_-" for character in compact) and any(
            character in "|/;:_-" for character in compact
        ):
            continue
        box = line["bounding_box"]
        center = (box["y"] + box["height"] / 2) / page["height"]
        if (
            stripped.isdigit()
            and len(stripped) <= 2
            and any(abs(center - schedule_center) <= 0.025 for schedule_center in schedule_centers)
        ):
            continue
        retained.append(line)
    page["lines"] = retained
    return original_count - len(retained)


def _reconcile_balance_sheet(
    page: dict[str, Any],
    alternatives: dict[int, dict[int, str]],
) -> int:
    lines = page["lines"]
    page_text = "\n".join(line["text"] for line in lines).casefold()
    if "balance sheet" not in page_text or "capital and liabilities" not in page_text:
        return 0

    corrections = 0
    for labels in (
        _BALANCE_SHEET_LIABILITY_LABELS,
        _BALANCE_SHEET_ASSET_LABELS,
    ):
        component_indices = [
            _find_financial_line(lines, label) for label in labels
        ]
        if any(index is None for index in component_indices):
            continue
        resolved_indices = [int(index) for index in component_indices]
        total_index = _find_total_line(lines, max(resolved_indices) + 1)
        if total_index is None:
            continue

        component_values = [_line_financial_values(lines[index]) for index in resolved_indices]
        total_values = _line_financial_values(lines[total_index])
        if any(len(values) < 2 for values in component_values) or len(total_values) < 2:
            continue

        for column in range(2):
            current_values = [values[-2 + column] for values in component_values]
            expected_total = total_values[-2 + column]
            difference = expected_total - sum(current_values)
            if difference == 0:
                continue

            for row_position, line_index in enumerate(resolved_indices):
                alternative_text = alternatives.get(line_index, {}).get(column)
                if alternative_text is None:
                    continue
                alternative_value = _parse_financial_number(alternative_text)
                if alternative_value - current_values[row_position] != difference:
                    continue
                current_text = _format_financial_number(current_values[row_position])
                updated = lines[line_index]["text"].replace(
                    current_text,
                    alternative_text,
                    1,
                )
                if updated != lines[line_index]["text"]:
                    lines[line_index]["text"] = updated
                    component_values[row_position][-2 + column] = alternative_value
                    corrections += 1
                break

    return corrections


def _find_financial_line(
    lines: list[dict[str, Any]],
    label: str,
) -> int | None:
    for index, line in enumerate(lines):
        text = line["text"].strip().casefold()
        if text.startswith(label) and len(_line_financial_values(line)) >= 2:
            return index
    return None


def _find_total_line(
    lines: list[dict[str, Any]],
    start_index: int,
) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index]["text"].strip().casefold().startswith("total"):
            if len(_line_financial_values(lines[index])) >= 2:
                return index
    return None


def _line_financial_values(line: dict[str, Any]) -> list[Decimal]:
    return [
        _parse_financial_number(value)
        for value in _GROUPED_FINANCIAL_NUMBER.findall(line["text"])
    ]


def _parse_financial_number(value: str) -> Decimal:
    return Decimal(value.replace(",", ""))


def _format_financial_number(value: Decimal) -> str:
    decimal_places = max(0, -value.as_tuple().exponent)
    return f"{value:,.{decimal_places}f}"


def _is_strict_financial_number(value: str) -> bool:
    return _STRICT_FINANCIAL_NUMBER.fullmatch(value.strip()) is not None


def _replace_flexible(text: str, original: str, replacement: str) -> str:
    tokens = original.split()
    if not tokens:
        return text
    pattern = r"\s*".join(re.escape(token) for token in tokens)
    return re.sub(pattern, lambda _: replacement, text, count=1)


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
        tolerance = max(6.0, min(box["height"], row_height) * 0.65)
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
        "multi_pass": _boolean_setting("OCR_MULTI_PASS", True),
    }
    if render_dpi is not None:
        details["render_dpi"] = render_dpi
        if details["multi_pass"]:
            details["secondary_render_dpi"] = _integer_setting(
                "OCR_SECONDARY_RENDER_DPI",
                DEFAULT_SECONDARY_OCR_RENDER_DPI,
                MIN_OCR_RENDER_DPI,
                MAX_OCR_RENDER_DPI,
            )
    if details["multi_pass"]:
        details["secondary_page_segmentation_mode"] = _integer_setting(
            "OCR_SECONDARY_PSM",
            DEFAULT_SECONDARY_TESSERACT_PSM,
            0,
            13,
        )
        details["sparse_page_segmentation_mode"] = DEFAULT_SPARSE_TESSERACT_PSM
    return details


def _integer_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _boolean_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}
