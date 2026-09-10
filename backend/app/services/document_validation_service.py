from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader


MAX_PDF_PAGES = 3
SUPPORTED_FILE_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}
EXPECTED_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}


def _result(
    file_name: str,
    file_type: str,
    *,
    is_supported: bool,
    is_readable: bool,
    page_count: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    file_validation: dict[str, Any] = {
        "file_type": file_type,
        "is_supported": is_supported,
        "is_readable": is_readable,
        "page_count": page_count,
        "status": "PASS" if is_supported and is_readable and error is None else "FAIL",
    }
    if error is not None:
        file_validation["error"] = error

    return {
        "file_name": file_name,
        "file_validation": file_validation,
    }


def validate_document(file_name: str, content: bytes) -> dict[str, Any]:
    """Validate a supported document without performing extraction."""
    suffix = Path(file_name).suffix.lower()
    file_type = SUPPORTED_FILE_TYPES.get(suffix, "application/octet-stream")

    if suffix not in SUPPORTED_FILE_TYPES:
        return _result(
            file_name,
            file_type,
            is_supported=False,
            is_readable=False,
            page_count=None,
            error="Unsupported file type. Allowed types are PDF, JPG/JPEG, and PNG.",
        )

    if not content:
        return _result(
            file_name,
            file_type,
            is_supported=True,
            is_readable=False,
            page_count=None,
            error="The uploaded file is empty.",
        )

    if suffix == ".pdf":
        return _validate_pdf(file_name, file_type, content)

    return _validate_image(file_name, file_type, suffix, content)


def _validate_pdf(
    file_name: str,
    file_type: str,
    content: bytes,
) -> dict[str, Any]:
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("The PDF is encrypted and cannot be read.")

        page_count = len(reader.pages)
        for page in reader.pages:
            _ = page.mediabox
    except Exception:
        return _result(
            file_name,
            file_type,
            is_supported=True,
            is_readable=False,
            page_count=None,
            error="The PDF is corrupted, encrypted, or unreadable.",
        )

    if page_count > MAX_PDF_PAGES:
        return _result(
            file_name,
            file_type,
            is_supported=True,
            is_readable=True,
            page_count=page_count,
            error=f"PDFs may contain at most {MAX_PDF_PAGES} pages.",
        )

    return _result(
        file_name,
        file_type,
        is_supported=True,
        is_readable=True,
        page_count=page_count,
    )


def _validate_image(
    file_name: str,
    file_type: str,
    suffix: str,
    content: bytes,
) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format != EXPECTED_IMAGE_FORMATS[suffix]:
                raise ValueError("The image content does not match its file extension.")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return _result(
            file_name,
            file_type,
            is_supported=True,
            is_readable=False,
            page_count=None,
            error="The image is corrupted, unreadable, or does not match its extension.",
        )

    return _result(
        file_name,
        file_type,
        is_supported=True,
        is_readable=True,
        page_count=None,
    )
