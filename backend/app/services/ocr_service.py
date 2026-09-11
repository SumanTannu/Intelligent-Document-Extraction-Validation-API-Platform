import os
import re
from copy import deepcopy
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
TESSERACT_WINDOWS_INSTALL_DIR = "Tesseract-OCR"
TESSERACT_WINDOWS_EXECUTABLE = "tesseract.exe"

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
_CURRENCY_UNIT_LINE = re.compile(
    r"^\s*[^\w\d]*\s*in\s+[^\dA-Za-z]*[O0]{3}\s*$",
    flags=re.IGNORECASE,
)
_COMMON_FINANCIAL_OCR_WORDS = {"dimunition": "diminution"}
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
    try:
        primary_page = _ocr_image(primary_image, page_number)
        if not _boolean_setting("OCR_MULTI_PASS", True):
            return primary_page

        sparse_pass_enabled = _boolean_setting("OCR_SPARSE_PASS", True)
        sparse_page = (
            _ocr_image(
                primary_image,
                page_number,
                psm_override=DEFAULT_SPARSE_TESSERACT_PSM,
            )
            if sparse_pass_enabled
            else None
        )
    finally:
        primary_image.close()

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
    # Render the second pass only after releasing the primary image. Re-rendering
    # at the same DPI costs a little CPU but avoids holding two full-page bitmaps.
    secondary_image = _render_pdf_page(page, secondary_dpi)
    try:
        secondary_page = _ocr_image(
            secondary_image,
            page_number,
            psm_override=secondary_psm,
        )
        return _enhance_ocr_page(
            primary_page,
            secondary_page,
            sparse_page or secondary_page,
            secondary_image,
            sparse_pass_enabled=sparse_pass_enabled,
        )
    finally:
        secondary_image.close()


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
    try:
        return _ocr_prepared_image(
            prepared_image,
            page_number,
            psm_override=psm_override,
        )
    finally:
        prepared_image.close()


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
    try:
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
        sparse_pass_enabled = _boolean_setting("OCR_SPARSE_PASS", True)
        sparse_page = (
            _ocr_prepared_image(
                prepared_image,
                page_number,
                psm_override=DEFAULT_SPARSE_TESSERACT_PSM,
            )
            if sparse_pass_enabled
            else secondary_page
        )
        return _enhance_ocr_page(
            primary_page,
            secondary_page,
            sparse_page,
            prepared_image,
            sparse_pass_enabled=sparse_pass_enabled,
        )
    finally:
        prepared_image.close()


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
    _configure_tesseract_command()

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


def _configure_tesseract_command() -> str:
    """Configure pytesseract from the environment or common Windows installs."""
    configured_command = os.getenv("TESSERACT_CMD", "").strip()
    if configured_command:
        pytesseract.pytesseract.tesseract_cmd = configured_command
        return configured_command

    for environment_variable in ("ProgramFiles", "ProgramFiles(x86)"):
        program_files = os.getenv(environment_variable, "").strip()
        if not program_files:
            continue

        candidate = (
            Path(program_files)
            / TESSERACT_WINDOWS_INSTALL_DIR
            / TESSERACT_WINDOWS_EXECUTABLE
        )
        if candidate.is_file():
            command = str(candidate)
            pytesseract.pytesseract.tesseract_cmd = command
            return command

    return pytesseract.pytesseract.tesseract_cmd


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
    *,
    sparse_pass_enabled: bool,
) -> dict[str, Any]:
    original_lines = deepcopy(primary_page["lines"])
    primary_page["source_lines"] = original_lines
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
    restored_substantive_rows = _restore_lost_substantive_rows(
        primary_page,
        original_lines,
    )
    reconstructed_financial_rows, table_regions = _reconstruct_financial_tables(
        primary_page
    )
    lost_substantive_rows = _lost_substantive_rows(
        primary_page,
        original_lines,
    )
    if lost_substantive_rows:
        raise TextExtractionError(
            "OCR normalization could not safely preserve every substantive table row."
        )
    primary_page["text"] = _join_page_text(
        [line["text"] for line in primary_page["lines"]]
    )
    primary_page["table_regions"] = table_regions
    primary_page["enhancement"] = {
        "multi_pass": True,
        "sparse_pass": sparse_pass_enabled,
        "numeric_replacements": numeric_replacements,
        "schedule_replacements": schedule_replacements,
        "text_replacements": text_replacements,
        "recovered_headings": heading_recoveries,
        "reconciled_values": reconciled_values,
        "common_text_corrections": common_text_corrections,
        "removed_layout_noise": removed_layout_noise,
        "restored_substantive_rows": restored_substantive_rows,
        "lost_substantive_rows": 0,
        "reconstructed_financial_rows": reconstructed_financial_rows,
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
    return replacements + _improve_spatial_text_segments(
        primary_page,
        secondary_page,
        sparse_page,
    )


def _improve_spatial_text_segments(
    primary_page: dict[str, Any],
    secondary_page: dict[str, Any],
    sparse_page: dict[str, Any],
) -> int:
    """Replace a column fragment only when a stronger pass occupies that column."""
    replacements = 0
    for primary_line in primary_page["lines"]:
        if len(_financial_number_groups(primary_line, primary_page["width"])) >= 2:
            continue
        primary_box = primary_line["bounding_box"]
        for alternate_page in (secondary_page, sparse_page):
            for alternate_line in _nearby_spatial_lines(
                primary_line, primary_page, alternate_page
            ):
                alternate_box = alternate_line["bounding_box"]
                primary_width_ratio = primary_box["width"] / primary_page["width"]
                alternate_width_ratio = alternate_box["width"] / alternate_page["width"]
                if alternate_width_ratio > primary_width_ratio * 0.6:
                    continue

                alternate_left = alternate_box["x"] / alternate_page["width"]
                alternate_right = (
                    alternate_box["x"] + alternate_box["width"]
                ) / alternate_page["width"]
                spatial_words = [
                    word
                    for word in primary_line.get("words", [])
                    if alternate_left - 0.015
                    <= _word_horizontal_center(word) / primary_page["width"]
                    <= alternate_right + 0.015
                ]
                if len(spatial_words) < 2:
                    continue
                spatial_words.sort(key=lambda word: word["bounding_box"]["x"])
                original = " ".join(word["text"] for word in spatial_words)
                replacement = alternate_line["text"].strip()
                original_key = re.sub(r"\W", "", original.casefold())
                replacement_key = re.sub(r"\W", "", replacement.casefold())
                if not original_key or not replacement_key:
                    continue
                if SequenceMatcher(None, original_key, replacement_key).ratio() < 0.82:
                    continue
                if _line_confidence(alternate_line) < _weighted_word_confidence(spatial_words) + 8:
                    continue
                updated = _replace_flexible(primary_line["text"], original, replacement)
                if updated != primary_line["text"]:
                    primary_line["text"] = updated
                    replacements += 1
    return replacements


def _nearby_spatial_lines(
    primary_line: dict[str, Any],
    primary_page: dict[str, Any],
    alternate_page: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_box = primary_line["bounding_box"]
    primary_center = (
        primary_box["y"] + primary_box["height"] / 2
    ) / primary_page["height"]
    matches = []
    for line in alternate_page["lines"]:
        box = line["bounding_box"]
        center = (box["y"] + box["height"] / 2) / alternate_page["height"]
        distance = abs(primary_center - center)
        if distance <= 0.018:
            matches.append((box["width"] / alternate_page["width"], distance, line))
    return [line for _, _, line in sorted(matches, key=lambda item: (item[0], item[1]))]


def _weighted_word_confidence(words: list[dict[str, Any]]) -> float:
    if not words:
        return 0.0
    weights = [max(len(word["text"]), 1) for word in words]
    return sum(
        word["confidence"] * weight for word, weight in zip(words, weights)
    ) / sum(weights)


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
        if _CURRENCY_UNIT_LINE.fullmatch(updated):
            updated = "₹ in '000"
        for incorrect, corrected in _COMMON_FINANCIAL_OCR_WORDS.items():
            updated = re.sub(
                rf"\b{re.escape(incorrect)}\b",
                corrected,
                updated,
                flags=re.IGNORECASE,
            )
        if updated != line["text"]:
            line["text"] = updated
            corrections += 1

    secondary_headers = [
        line
        for line in secondary_page["lines"]
        if line["text"].strip().casefold().startswith("schedule")
        and len(_SHORT_TABLE_DATE.findall(line["text"])) >= 2
    ]
    for secondary_header in secondary_headers:
        primary_header = _nearest_line(
            secondary_header,
            primary_page["lines"],
            secondary_page["height"],
            primary_page["height"],
        )
        if primary_header is None:
            continue
        if not primary_header["text"].strip().casefold().startswith("schedule"):
            continue
        if _is_substantive_table_row(primary_header, primary_page["width"]):
            continue

        secondary_dates = _SHORT_TABLE_DATE.findall(secondary_header["text"])
        recovered_header = (
            f"Schedule        As at {secondary_dates[0]}"
            f"        As at {secondary_dates[1]}"
        )
        if primary_header["text"] != recovered_header:
            primary_header["text"] = recovered_header
            corrections += 1
    return corrections


def _restore_lost_substantive_rows(
    page: dict[str, Any],
    original_lines: list[dict[str, Any]],
) -> int:
    restored = 0
    for original in original_lines:
        if not _is_substantive_table_row(original, page["width"]):
            continue

        current = _line_at_same_position(original, page["lines"], page["height"])
        if current is not None and _retains_label_identity(original, current):
            continue

        restored_line = deepcopy(original)
        restored_line["text"] = normalize_financial_numbers(
            restored_line["raw_text"]
        )
        restored_line["normalization_restored"] = True
        if current is None:
            page["lines"].append(restored_line)
        else:
            page["lines"][page["lines"].index(current)] = restored_line
        restored += 1

    page["lines"].sort(
        key=lambda line: (
            line["bounding_box"]["y"],
            line["bounding_box"]["x"],
        )
    )
    return restored


def _lost_substantive_rows(
    page: dict[str, Any],
    original_lines: list[dict[str, Any]],
) -> list[str]:
    lost = []
    for original in original_lines:
        if not _is_substantive_table_row(original, page["width"]):
            continue
        current = _line_at_same_position(original, page["lines"], page["height"])
        if current is None or not _retains_label_identity(original, current):
            lost.append(original["raw_text"])
    return lost


def _is_substantive_table_row(line: dict[str, Any], page_width: int) -> bool:
    words = line.get("words", [])
    numeric_words = [
        word
        for word in words
        if _is_financial_value_token(word["text"])
        and _word_horizontal_center(word) >= page_width * 0.55
    ]
    if not numeric_words:
        return False

    first_value_x = min(word["bounding_box"]["x"] for word in numeric_words)
    label_words = [
        word["text"]
        for word in words
        if word["bounding_box"]["x"] < first_value_x
        and re.search(r"[A-Za-z]{2,}", word["text"])
    ]
    return len(label_words) >= 2


def _is_financial_value_token(text: str) -> bool:
    stripped = text.strip()
    normalized = _normalize_financial_number_candidate(stripped.strip("()"))
    return _is_strict_financial_number(normalized) or stripped in {"-", "–", "—"}


def _word_horizontal_center(word: dict[str, Any]) -> float:
    box = word["bounding_box"]
    return box["x"] + box["width"] / 2


def _line_at_same_position(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    page_height: int,
) -> dict[str, Any] | None:
    source_box = source["bounding_box"]
    source_center = source_box["y"] + source_box["height"] / 2
    matches = []
    for candidate in candidates:
        candidate_box = candidate["bounding_box"]
        candidate_center = candidate_box["y"] + candidate_box["height"] / 2
        distance = abs(source_center - candidate_center) / page_height
        if distance <= 0.008:
            source_horizontal_center = source_box["x"] + source_box["width"] / 2
            candidate_horizontal_center = (
                candidate_box["x"] + candidate_box["width"] / 2
            )
            horizontal_distance = abs(
                source_horizontal_center - candidate_horizontal_center
            ) / max(source_box["width"], candidate_box["width"], 1)
            label_mismatch = 0 if _retains_label_identity(source, candidate) else 1
            matches.append(
                (label_mismatch, distance, horizontal_distance, candidate)
            )
    if not matches:
        return None
    return min(matches, key=lambda match: match[:3])[3]


def _retains_label_identity(
    original: dict[str, Any],
    processed: dict[str, Any],
) -> bool:
    original_tokens = _label_tokens(original["raw_text"])
    if not original_tokens:
        return True
    processed_tokens = _label_tokens(processed["text"])
    retained = sum(token in processed_tokens for token in original_tokens)
    return retained / len(original_tokens) >= 0.6


def _label_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{3,}", text)
    }


def _financial_cells(
    line: dict[str, Any],
    page_width: int,
) -> list[dict[str, Any]]:
    value_words = [
        word
        for word in sorted(
            line.get("words", []), key=lambda item: item["bounding_box"]["x"]
        )
        if _word_horizontal_center(word) >= page_width * 0.55
        and _is_financial_value_token(word["text"])
    ]
    groups: list[list[dict[str, Any]]] = []
    for word in value_words:
        if not groups:
            groups.append([word])
            continue
        previous_box = groups[-1][-1]["bounding_box"]
        current_box = word["bounding_box"]
        gap = current_box["x"] - (previous_box["x"] + previous_box["width"])
        if gap <= page_width * 0.025:
            groups[-1].append(word)
        else:
            groups.append([word])

    cells = []
    for group in groups:
        left = min(word["bounding_box"]["x"] for word in group)
        right = max(
            word["bounding_box"]["x"] + word["bounding_box"]["width"]
            for word in group
        )
        raw_value = " ".join(word["text"] for word in group)
        cells.append(
            {
                "raw_text": raw_value,
                "text": _normalize_financial_number_candidate(raw_value),
                "x": left,
                "right": right,
                "center_x": (left + right) / 2,
                "words": group,
            }
        )
    return cells


def _reconstruct_financial_tables(
    page: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Join wrapped labels to numeric rows and expose coordinate-derived cells."""
    lines = page["lines"]
    reconstructed = 0
    index = 0
    while index < len(lines):
        cells = _financial_cells(lines[index], page["width"])
        if len(cells) < 2 or _line_has_label_left_of_values(lines[index], cells):
            index += 1
            continue

        label_indices = []
        if index > 0 and _is_adjacent_wrapped_label(
            lines[index - 1], lines[index], page
        ):
            label_indices.append(index - 1)
        if index + 1 < len(lines) and _is_adjacent_wrapped_label(
            lines[index + 1], lines[index], page
        ):
            label_indices.append(index + 1)
        if not label_indices:
            index += 1
            continue

        selected = sorted(set(label_indices + [index]))
        label_lines = [lines[item] for item in selected if item != index]
        label_lines.sort(key=lambda item: item["bounding_box"]["y"])
        label_text = " ".join(item["text"].strip() for item in label_lines)
        value_text = "        ".join(cell["text"] for cell in cells)
        source_lines = [deepcopy(lines[item]) for item in selected]
        all_words = [
            word for item in selected for word in lines[item].get("words", [])
        ]
        left = min(item["bounding_box"]["x"] for item in source_lines)
        top = min(item["bounding_box"]["y"] for item in source_lines)
        right = max(
            item["bounding_box"]["x"] + item["bounding_box"]["width"]
            for item in source_lines
        )
        bottom = max(
            item["bounding_box"]["y"] + item["bounding_box"]["height"]
            for item in source_lines
        )
        combined = {
            "raw_text": "\n".join(item["raw_text"] for item in source_lines),
            "text": f"{label_text}                {value_text}",
            "bounding_box": {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            },
            "words": sorted(all_words, key=_word_position),
            "source_lines": source_lines,
            "reconstructed_label": label_text,
        }
        first = selected[0]
        for item in reversed(selected):
            del lines[item]
        lines.insert(first, combined)
        reconstructed += 1
        index = first + 1

    table_rows = []
    for line in lines:
        cells = _financial_cells(line, page["width"])
        if not cells:
            continue
        label_words = [
            word
            for word in line.get("words", [])
            if word["bounding_box"]["x"] < min(cell["x"] for cell in cells)
        ]
        if not any(re.search(r"[A-Za-z]", word["text"]) for word in label_words):
            continue
        label_words.sort(key=lambda word: word["bounding_box"]["x"])
        label = line.get("reconstructed_label") or " ".join(
            word["text"] for word in label_words
        )
        row = {
            "label": label,
            "values": [cell["text"] for cell in cells],
            "value_columns": [cell["center_x"] for cell in cells],
            "bounding_box": deepcopy(line["bounding_box"]),
        }
        line["table_row"] = row
        table_rows.append(row)

    return reconstructed, _group_table_regions(
        table_rows, page["width"], page["height"]
    )


def _line_has_label_left_of_values(
    line: dict[str, Any],
    cells: list[dict[str, Any]],
) -> bool:
    first_value_x = min(cell["x"] for cell in cells)
    return any(
        word["bounding_box"]["x"] < first_value_x
        and re.search(r"[A-Za-z]{2,}", word["text"])
        for word in line.get("words", [])
    )


def _is_adjacent_wrapped_label(
    candidate: dict[str, Any],
    value_line: dict[str, Any],
    page: dict[str, Any],
) -> bool:
    if _financial_cells(candidate, page["width"]):
        return False
    if not any(
        re.search(r"[A-Za-z]{2,}", word["text"])
        and _word_horizontal_center(word) < page["width"] * 0.65
        for word in candidate.get("words", [])
    ):
        return False
    candidate_box = candidate["bounding_box"]
    value_box = value_line["bounding_box"]
    candidate_center = candidate_box["y"] + candidate_box["height"] / 2
    value_center = value_box["y"] + value_box["height"] / 2
    return abs(candidate_center - value_center) / page["height"] <= 0.035


def _group_table_regions(
    rows: list[dict[str, Any]],
    page_width: int,
    page_height: int,
) -> list[dict[str, Any]]:
    regions: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: item["bounding_box"]["y"]):
        if not regions:
            regions.append([row])
            continue
        previous = regions[-1][-1]["bounding_box"]
        gap = row["bounding_box"]["y"] - (previous["y"] + previous["height"])
        if gap / page_height <= 0.05:
            regions[-1].append(row)
        else:
            regions.append([row])
    return [
        {
            "rows": region,
            "numeric_columns": _cluster_numeric_columns(region, page_width),
        }
        for region in regions
    ]


def _cluster_numeric_columns(
    rows: list[dict[str, Any]],
    scale: int,
) -> list[float]:
    centers = sorted(center for row in rows for center in row["value_columns"])
    clusters: list[list[float]] = []
    for center in centers:
        if not clusters or abs(center - median(clusters[-1])) / max(scale, 1) > 0.06:
            clusters.append([center])
        else:
            clusters[-1].append(center)
    return [round(median(cluster), 2) for cluster in clusters]


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
        if compact and all(character in "iIl|/;:_.-" for character in compact) and any(
            character in "|/;:_.-" for character in compact
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
        if _is_probable_duplicate_logo_line(line, page["lines"], page["height"]):
            continue
        retained.append(line)
    page["lines"] = retained
    return original_count - len(retained)


def _is_probable_duplicate_logo_line(
    line: dict[str, Any],
    lines: list[dict[str, Any]],
    page_height: int,
) -> bool:
    """Drop low-confidence footer logo glyphs duplicated by nearby footer text."""
    box = line["bounding_box"]
    if box["y"] / page_height < 0.75 or len(line.get("words", [])) > 4:
        return False
    alpha_tokens = {
        re.sub(r"[^a-z]", "", word["text"].casefold())
        for word in line.get("words", [])
    }
    alpha_tokens.discard("")
    if len(alpha_tokens) < 2:
        return False
    for other in lines:
        if other is line or other["bounding_box"]["y"] <= box["y"]:
            continue
        if (other["bounding_box"]["y"] - box["y"]) / page_height > 0.05:
            continue
        other_tokens = {
            re.sub(r"[^a-z]", "", word["text"].casefold())
            for word in other.get("words", [])
        }
        other_tokens.discard("")
        looks_like_graphic_text = (
            _line_confidence(line) < 90
            or box["height"] > other["bounding_box"]["height"] * 1.3
            or any(
                not re.search(r"[A-Za-z]", word["text"])
                for word in line.get("words", [])
            )
        )
        if len(alpha_tokens & other_tokens) >= 2 and looks_like_graphic_text:
            return True
    return False


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
        "segments": _spatial_segments(words, typical_character_width),
    }


def _spatial_segments(
    words: list[dict[str, Any]],
    typical_character_width: float,
) -> list[dict[str, Any]]:
    """Retain coordinate-separated columns instead of only flattening to text."""
    groups: list[list[dict[str, Any]]] = []
    for word in words:
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]["bounding_box"]
        current = word["bounding_box"]
        gap = current["x"] - (previous["x"] + previous["width"])
        if gap > max(12, typical_character_width * 3.5):
            groups.append([word])
        else:
            groups[-1].append(word)
    segments = []
    for group in groups:
        left = min(word["bounding_box"]["x"] for word in group)
        top = min(word["bounding_box"]["y"] for word in group)
        right = max(
            word["bounding_box"]["x"] + word["bounding_box"]["width"]
            for word in group
        )
        bottom = max(
            word["bounding_box"]["y"] + word["bounding_box"]["height"]
            for word in group
        )
        segments.append(
            {
                "text": " ".join(word["text"] for word in group),
                "bounding_box": {
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                },
                "confidence": _weighted_word_confidence(group),
            }
        )
    return segments


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
    details["sparse_pass"] = (
        details["multi_pass"] and _boolean_setting("OCR_SPARSE_PASS", True)
    )
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
        if details["sparse_pass"]:
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
