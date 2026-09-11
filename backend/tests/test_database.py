from __future__ import annotations

from collections.abc import Generator
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.document_service as document_service
from app.core.database import Base, get_db, init_db
from app.main import app
from app.models.document import Document
from app.repositories.document_repository import (
    DocumentPersistenceError,
    DuplicateDocumentError,
    create_document_result,
    document_to_processing_result,
    get_document_by_name,
    list_documents,
)
from app.schemas.document import DocumentProcessResponse
from app.schemas.extraction import (
    FinancialValidationCheck,
    FinancialValidationResult,
    InvoiceExtraction,
)
from app.services.ocr_service import TextExtractionError


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session(test_engine: Engine) -> Generator[Session, None, None]:
    factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    with factory() as database_session:
        yield database_session


@pytest.fixture
def api_database(
    test_engine: Engine,
) -> Generator[sessionmaker[Session], None, None]:
    factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )

    def override_get_db() -> Generator[Session, None, None]:
        with factory() as database_session:
            yield database_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield factory
    finally:
        app.dependency_overrides.pop(get_db, None)


def sample_processing_result(
    file_name: str = "invoice-825.pdf",
) -> dict[str, Any]:
    structured_extraction = InvoiceExtraction(
        document_type="invoice",
        extracted_fields=[
            {
                "field": "total_amount",
                "value": "$157.48",
                "evidence": [
                    {
                        "source_text": "Total Due $157.48",
                        "page_number": 1,
                        "bounding_box": {
                            "x": 998.0,
                            "y": 899.0,
                            "width": 220.0,
                            "height": 42.0,
                        },
                    }
                ],
                "confidence": 0.97,
            }
        ],
    )
    financial_validation = FinancialValidationResult(
        document_type="invoice",
        tolerance=Decimal("0.01"),
        status="PASS",
        checks=[
            FinancialValidationCheck(
                rule_name="invoice_tax_reconciliation",
                formula="taxable_amount + tax = total_amount",
                input_values={
                    "taxable_amount": Decimal("145.00"),
                    "tax": Decimal("12.48"),
                },
                calculated_value=Decimal("157.48"),
                reported_value=Decimal("157.48"),
                variance=Decimal("0.00"),
                status="PASS",
            )
        ],
    )
    return {
        "file_name": file_name,
        "document_type": "invoice",
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
        "text_extraction": {
            "extracted_text": "Invoice 825\nTotal Due $157.48",
            "extraction_method": "native_text",
            "page_count": 1,
            "status": "PASS",
            "raw_text": "Invoice 825\nTotal Due $157.48",
            "pages": [
                {
                    "page_number": 1,
                    "text": "Invoice 825\nTotal Due $157.48",
                }
            ],
        },
        "structured_extraction": structured_extraction,
        "financial_validation": financial_validation,
    }


def stub_completed_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = sample_processing_result()
    monkeypatch.setattr(
        document_service,
        "validate_document",
        lambda file_name, _content: {
            "file_name": file_name,
            "file_validation": payload["file_validation"],
        },
    )
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda _file_name, _content: payload["text_extraction"],
    )
    monkeypatch.setattr(
        document_service,
        "extract_structured_data",
        lambda **_kwargs: payload["structured_extraction"],
    )
    monkeypatch.setattr(
        document_service,
        "validate_financial_data",
        lambda _extraction: payload["financial_validation"],
    )


def test_init_db_creates_documents_table(test_engine: Engine) -> None:
    assert "documents" in inspect(test_engine).get_table_names()


def test_create_and_retrieve_document_by_name(session: Session) -> None:
    payload = sample_processing_result()

    created = create_document_result(session, payload)
    retrieved = get_document_by_name(session, "invoice-825.pdf")

    assert created.id is not None
    assert created.processing_status == "COMPLETED"
    assert retrieved is not None
    assert retrieved.id == created.id


def test_list_documents_uses_deterministic_creation_order(session: Session) -> None:
    first = create_document_result(session, sample_processing_result("first.pdf"))
    second = create_document_result(session, sample_processing_result("second.pdf"))

    documents = list_documents(session)

    assert [document.id for document in documents] == [first.id, second.id]
    assert [document.file_name for document in documents] == [
        "first.pdf",
        "second.pdf",
    ]


def test_complete_processing_result_round_trip_preserves_all_sections(
    session: Session,
) -> None:
    payload = sample_processing_result()
    expected = DocumentProcessResponse.model_validate(payload).model_dump(
        mode="json",
        exclude_none=True,
    )

    stored = create_document_result(session, payload)
    restored = document_to_processing_result(stored).model_dump(
        mode="json",
        exclude_none=True,
    )

    assert restored == expected
    evidence = restored["structured_extraction"]["extracted_fields"][0]
    assert evidence["confidence"] == 0.97
    assert evidence["evidence"][0]["source_text"] == "Total Due $157.48"
    assert evidence["evidence"][0]["bounding_box"]["x"] == 998.0
    assert restored["financial_validation"]["checks"][0]["status"] == "PASS"
    assert restored["text_extraction"]["extraction_method"] == "native_text"
    assert restored["file_validation"]["status"] == "PASS"


def test_duplicate_document_name_is_rejected_without_overwrite(
    session: Session,
) -> None:
    original = sample_processing_result()
    create_document_result(session, original)

    duplicate = sample_processing_result()
    duplicate["text_extraction"] = {
        **duplicate["text_extraction"],
        "extracted_text": "replacement text",
    }

    with pytest.raises(DuplicateDocumentError):
        create_document_result(session, duplicate)

    documents = list_documents(session)
    assert len(documents) == 1
    assert documents[0].text_extraction_json["extracted_text"] != "replacement text"


def test_database_failure_rolls_back() -> None:
    class FailingSession:
        rollback_called = False

        def scalar(self, _statement: object) -> None:
            return None

        def add(self, _document: object) -> None:
            pass

        def commit(self) -> None:
            raise SQLAlchemyError("simulated commit failure")

        def refresh(self, _document: object) -> None:
            raise AssertionError("Refresh must not happen after a failed commit.")

        def rollback(self) -> None:
            self.rollback_called = True

    failing_session = FailingSession()

    with pytest.raises(DocumentPersistenceError):
        create_document_result(failing_session, sample_processing_result())  # type: ignore[arg-type]

    assert failing_session.rollback_called is True


def test_successful_post_persists_and_get_endpoints_return_result(
    api_database: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_completed_processing(monkeypatch)
    client = TestClient(app)

    post_response = client.post(
        "/api/v1/documents/process",
        files={"file": ("invoice-825.pdf", b"document", "application/pdf")},
        data={"document_type": "invoice"},
    )
    list_response = client.get("/api/v1/documents")
    detail_response = client.get("/api/v1/documents/invoice-825.pdf")

    assert post_response.status_code == 202
    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert list_response.json()["documents"] == [post_response.json()]
    assert detail_response.json() == post_response.json()
    with api_database() as verification_session:
        assert verification_session.scalar(
            select(func.count()).select_from(Document)
        ) == 1


def test_duplicate_filename_returns_http_409(
    api_database: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_completed_processing(monkeypatch)
    client = TestClient(app)
    request = {
        "files": {"file": ("invoice-825.pdf", b"document", "application/pdf")},
        "data": {"document_type": "invoice"},
    }

    assert client.post("/api/v1/documents/process", **request).status_code == 202
    response = client.post("/api/v1/documents/process", **request)

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    with api_database() as verification_session:
        assert verification_session.scalar(
            select(func.count()).select_from(Document)
        ) == 1


def test_failed_validation_does_not_persist_a_document(
    api_database: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        document_service,
        "validate_document",
        lambda file_name, _content: {
            "file_name": file_name,
            "file_validation": {
                "file_type": "application/pdf",
                "is_supported": True,
                "is_readable": False,
                "page_count": None,
                "status": "FAIL",
            },
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("broken.pdf", b"broken", "application/pdf")},
        data={"document_type": "invoice"},
    )

    assert response.status_code == 400
    with api_database() as verification_session:
        assert verification_session.scalar(
            select(func.count()).select_from(Document)
        ) == 0


def test_failed_text_extraction_does_not_persist_a_document(
    api_database: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = sample_processing_result()
    monkeypatch.setattr(
        document_service,
        "validate_document",
        lambda file_name, _content: {
            "file_name": file_name,
            "file_validation": payload["file_validation"],
        },
    )
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda _file_name, _content: (_ for _ in ()).throw(
            TextExtractionError("Controlled extraction failure.")
        ),
    )

    response = TestClient(app).post(
        "/api/v1/documents/process",
        files={"file": ("unreadable.pdf", b"document", "application/pdf")},
        data={"document_type": "invoice"},
    )

    assert response.status_code == 422
    with api_database() as verification_session:
        assert verification_session.scalar(
            select(func.count()).select_from(Document)
        ) == 0


def test_database_failure_returns_sanitized_http_503(
    api_database: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_completed_processing(monkeypatch)
    monkeypatch.setattr(
        document_service,
        "create_document_result",
        lambda _session, _result: (_ for _ in ()).throw(
            DocumentPersistenceError("internal database path and details")
        ),
    )

    response = TestClient(app).post(
        "/api/v1/documents/process",
        files={"file": ("invoice-825.pdf", b"document", "application/pdf")},
        data={"document_type": "invoice"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The processed document could not be persisted."
    }
    assert "internal database path" not in response.text


def test_get_unknown_document_returns_404(
    api_database: sessionmaker[Session],
) -> None:
    response = TestClient(app).get("/api/v1/documents/unknown.pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document 'unknown.pdf' was not found."
