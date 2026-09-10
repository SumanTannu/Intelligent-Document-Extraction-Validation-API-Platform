from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status


DocumentType = Literal[
    "invoice",
    "balance_sheet",
    "profit_and_loss",
    "cash_flow_statement",
]

router = APIRouter()


@router.get("/health", tags=["health"], summary="Check service health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.post(
    "/documents/process",
    tags=["documents"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept a document for processing",
)
async def process_document(
    file: Annotated[UploadFile, File(description="Document to process")],
    document_type: Annotated[
        DocumentType,
        Form(description="Type of financial document"),
    ],
) -> dict[str, str]:
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )

    return {
        "status": "accepted",
        "message": "Document processing is not implemented in Step 2.",
        "document_name": file.filename,
        "document_type": document_type,
    }


@router.get(
    "/documents",
    tags=["documents"],
    summary="List processed documents",
)
async def list_documents() -> dict[str, list[object]]:
    return {"documents": []}


@router.get(
    "/documents/{document_name}",
    tags=["documents"],
    summary="Get a processed document by name",
)
async def get_document(document_name: str) -> dict[str, str]:
    if not document_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document name must not be blank.",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document '{document_name}' was not found.",
    )
