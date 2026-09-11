from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from app.schemas.extraction import (
    DocumentType,
    FinancialValidationResult,
    StructuredExtraction,
)


AnnotatedFileName = Annotated[StrictStr, Field(min_length=1)]
AnnotatedMimeType = Annotated[StrictStr, Field(min_length=1)]
AnnotatedPageCount = Annotated[int, Field(ge=1)]


class DocumentResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FileValidationResponse(DocumentResponseModel):
    file_type: AnnotatedMimeType
    is_supported: StrictBool
    is_readable: StrictBool
    page_count: AnnotatedPageCount | None = None
    status: Literal["PASS", "FAIL"]


class TextExtractionResponse(DocumentResponseModel):
    extracted_text: str
    extraction_method: Literal["native_text", "ocr"] | None
    page_count: AnnotatedPageCount | None
    status: Literal["PASS", "FAIL"]
    error: str | None = None
    raw_text: str | None = None
    pages: list[dict[str, Any]] | None = None
    ocr_details: dict[str, Any] | None = None


class DocumentProcessResponse(DocumentResponseModel):
    file_name: AnnotatedFileName
    document_type: DocumentType
    file_validation: FileValidationResponse
    text_extraction: TextExtractionResponse
    structured_extraction: StructuredExtraction | None = None
    financial_validation: FinancialValidationResult | None = None


class DocumentListResponse(DocumentResponseModel):
    documents: list[DocumentProcessResponse]
