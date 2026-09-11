from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.document import DocumentProcessResponse


logger = logging.getLogger(__name__)


class DocumentRepositoryError(Exception):
    """Base error for controlled persistence and retrieval failures."""


class DuplicateDocumentError(DocumentRepositoryError):
    """Raised when a document name already exists."""


class DocumentPersistenceError(DocumentRepositoryError):
    """Raised when a database operation cannot be completed."""


def create_document_result(
    session: Session,
    processing_result: dict[str, Any],
) -> Document:
    """Persist one complete processing result in a single transaction."""
    try:
        validated_result = DocumentProcessResponse.model_validate(processing_result)
        file_name = validated_result.file_name
        existing = session.scalar(
            select(Document.id).where(Document.file_name == file_name)
        )
        if existing is not None:
            raise DuplicateDocumentError(
                f"A document named '{file_name}' already exists."
            )

        document = Document(
            file_name=file_name,
            document_type=validated_result.document_type,
            processing_status="COMPLETED",
            file_validation_json=_to_json_compatible(
                validated_result.file_validation
            ),
            text_extraction_json=_to_json_compatible(
                validated_result.text_extraction
            ),
            structured_extraction_json=_optional_json_section(
                validated_result.structured_extraction
            ),
            financial_validation_json=_optional_json_section(
                validated_result.financial_validation
            ),
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        return document
    except DuplicateDocumentError:
        session.rollback()
        raise
    except IntegrityError as exc:
        session.rollback()
        logger.info("Document persistence rejected a duplicate filename.")
        raise DuplicateDocumentError(
            f"A document named '{file_name}' already exists."
        ) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Document persistence failed.")
        raise DocumentPersistenceError(
            "The processed document could not be persisted."
        ) from exc
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        session.rollback()
        logger.exception("The processing result could not be serialized safely.")
        raise DocumentPersistenceError(
            "The processed document could not be persisted."
        ) from exc


def list_documents(session: Session) -> list[Document]:
    """Return all persisted documents in deterministic creation order."""
    try:
        return list(
            session.scalars(select(Document).order_by(Document.id.asc())).all()
        )
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Document listing failed.")
        raise DocumentPersistenceError(
            "Persisted documents could not be retrieved."
        ) from exc


def get_document_by_name(
    session: Session,
    document_name: str,
) -> Document | None:
    """Return the unique document matching a filename, if one exists."""
    try:
        return session.scalar(
            select(Document).where(Document.file_name == document_name)
        )
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Document retrieval failed.")
        raise DocumentPersistenceError(
            "The persisted document could not be retrieved."
        ) from exc


def document_to_processing_result(
    document: Document,
) -> DocumentProcessResponse:
    """Reconstruct and validate the public processing response."""
    payload = {
        "file_name": document.file_name,
        "document_type": document.document_type,
        "file_validation": document.file_validation_json,
        "text_extraction": document.text_extraction_json,
        "structured_extraction": document.structured_extraction_json,
        "financial_validation": document.financial_validation_json,
    }
    # JSON-mode validation reconstructs Decimal/date values stored as JSON
    # while still enforcing the existing strict response contract.
    try:
        return DocumentProcessResponse.model_validate_json(json.dumps(payload))
    except (TypeError, ValueError, ValidationError) as exc:
        logger.exception("A persisted document result is invalid.")
        raise DocumentPersistenceError(
            "The persisted document could not be retrieved."
        ) from exc


def _optional_json_section(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result = _to_json_compatible(value)
    if not isinstance(result, dict):
        raise TypeError("A persisted result section must be a JSON object.")
    return result


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value
