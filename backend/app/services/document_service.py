from typing import Any

from sqlalchemy.orm import Session

from app.repositories.document_repository import create_document_result
from app.schemas.extraction import DocumentType
from app.services.document_validation_service import validate_document
from app.services.extraction_service import (
    StructuredExtractionError,
    extract_structured_data,
)
from app.services.financial_validation_service import validate_financial_data
from app.services.ocr_service import TextExtractionError, extract_text


class DocumentValidationError(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Document validation failed.")
        self.result = result


class DocumentExtractionError(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Document text extraction failed.")
        self.result = result


class DocumentStructuredExtractionError(Exception):
    def __init__(
        self,
        result: dict[str, Any],
        *,
        status_code: int,
    ) -> None:
        super().__init__("Document structured extraction failed.")
        self.result = result
        self.status_code = status_code


def process_document(
    file_name: str,
    content: bytes,
    document_type: DocumentType,
    session: Session,
) -> dict[str, Any]:
    """Process a document fully, then persist its complete result."""
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
                "document_type": document_type,
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

    try:
        structured_extraction = extract_structured_data(
            document_type=document_type,
            extracted_text=text_extraction["extracted_text"],
            pages=text_extraction.get("pages"),
        )
    except StructuredExtractionError as exc:
        raise DocumentStructuredExtractionError(
            {
                "file_name": file_name,
                "document_type": document_type,
                "file_validation": file_validation,
                "text_extraction": text_extraction,
                "structured_extraction": {
                    "status": "FAIL",
                    "error_code": exc.code,
                    "error": str(exc),
                },
            },
            status_code=_structured_extraction_status_code(exc.code),
        ) from exc

    financial_validation = validate_financial_data(structured_extraction)

    processing_result = {
        "file_name": file_name,
        "document_type": document_type,
        "file_validation": file_validation,
        "text_extraction": text_extraction,
        "structured_extraction": structured_extraction,
        "financial_validation": financial_validation,
    }
    create_document_result(session, processing_result)
    return processing_result


def _structured_extraction_status_code(error_code: str) -> int:
    if error_code == "structured_extraction_rate_limited":
        return 429
    if error_code in {
        "structured_extraction_not_configured",
        "structured_extraction_connection_failed",
    }:
        return 503
    if error_code == "structured_extraction_input_invalid":
        return 422
    return 502
