import logging
from time import monotonic
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.document_repository import (
    DocumentPersistenceError,
    DuplicateDocumentError,
    document_to_processing_result,
    get_document_by_name as get_document_by_name_from_database,
    list_documents as list_documents_from_database,
)
from app.schemas.document import DocumentListResponse, DocumentProcessResponse
from app.schemas.extraction import DocumentType
from app.services.document_service import (
    DocumentExtractionError,
    DocumentStructuredExtractionError,
    DocumentValidationError,
    process_document as process_document_file,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", tags=["health"], summary="Check service health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.post(
    "/documents/process",
    tags=["documents"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept a document for processing",
    response_model=DocumentProcessResponse,
    response_model_exclude_none=True,
)
async def process_document(
    file: Annotated[UploadFile, File(description="Document to process")],
    document_type: Annotated[
        DocumentType,
        Form(description="Type of financial document"),
    ],
    session: Annotated[Session, Depends(get_db)],
) -> DocumentProcessResponse:
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )

    content = await file.read()
    started_at = monotonic()
    logger.info("Document processing started.")
    try:
        result = await run_in_threadpool(
            process_document_file,
            file.filename,
            content,
            document_type,
            session,
        )
        logger.info(
            "Document processing completed in %.2f seconds.",
            monotonic() - started_at,
        )
        return result
    except DocumentValidationError as exc:
        logger.warning(
            "Document validation failed after %.2f seconds.",
            monotonic() - started_at,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.result,
        ) from None
    except DocumentExtractionError as exc:
        logger.warning(
            "Document text extraction failed after %.2f seconds.",
            monotonic() - started_at,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.result,
        ) from None
    except DocumentStructuredExtractionError as exc:
        logger.warning(
            "Document structured extraction failed after %.2f seconds.",
            monotonic() - started_at,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.result,
        ) from None
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from None
    except DocumentPersistenceError:
        logger.error("A processed document could not be persisted.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The processed document could not be persisted.",
        ) from None


@router.get(
    "/documents",
    tags=["documents"],
    summary="List processed documents",
    response_model=DocumentListResponse,
    response_model_exclude_none=True,
)
def list_documents(
    session: Annotated[Session, Depends(get_db)],
) -> DocumentListResponse:
    try:
        documents = list_documents_from_database(session)
        return DocumentListResponse(
            documents=[
                document_to_processing_result(document) for document in documents
            ]
        )
    except DocumentPersistenceError:
        logger.error("Persisted documents could not be listed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persisted documents could not be retrieved.",
        ) from None


@router.get(
    "/documents/{document_name}",
    tags=["documents"],
    summary="Get a processed document by name",
    response_model=DocumentProcessResponse,
    response_model_exclude_none=True,
)
def get_document(
    document_name: str,
    session: Annotated[Session, Depends(get_db)],
) -> DocumentProcessResponse:
    if not document_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document name must not be blank.",
        )

    try:
        document = get_document_by_name_from_database(session, document_name)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_name}' was not found.",
            )
        return document_to_processing_result(document)
    except DocumentPersistenceError:
        logger.error("A persisted document could not be retrieved.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The persisted document could not be retrieved.",
        ) from None
