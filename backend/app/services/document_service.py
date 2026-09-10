from typing import Any

from app.services.document_validation_service import validate_document
from app.services.ocr_service import TextExtractionError, extract_text


class DocumentValidationError(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Document validation failed.")
        self.result = result


class DocumentExtractionError(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Document text extraction failed.")
        self.result = result


def process_document(file_name: str, content: bytes) -> dict[str, Any]:
    """Validate a document, then extract its text."""
    validation_result = validate_document(file_name, content)
    file_validation = validation_result["file_validation"]

    if file_validation["status"] == "FAIL":
        raise DocumentValidationError(validation_result)

    try:
        text_extraction = extract_text(file_name, content)
    except TextExtractionError as exc:
        raise DocumentExtractionError(
            {
                "file_name": file_name,
                "file_validation": file_validation,
                "text_extraction": {
                    "extracted_text": "",
                    "extraction_method": None,
                    "page_count": file_validation["page_count"],
                    "status": "FAIL",
                    "error": str(exc),
                },
            }
        ) from exc

    return {
        "file_name": file_name,
        "file_validation": file_validation,
        "text_extraction": text_extraction,
    }
