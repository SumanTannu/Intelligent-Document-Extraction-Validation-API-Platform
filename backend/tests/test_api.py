from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from threading import Event
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from pydantic import TypeAdapter, ValidationError

import app.services.document_service as document_service
import app.services.extraction_service as extraction_service
from app.main import app
from app.schemas.extraction import (
    BalanceSheetExtraction,
    CashFlowStatementExtraction,
    InvoiceExtraction,
    ProfitAndLossExtraction,
    StructuredExtraction,
)
from app.services.extraction_service import (
    MAX_EXTRACTED_TEXT_CHARACTERS,
    StructuredExtractionAuthenticationError,
    StructuredExtractionConfigurationError,
    StructuredExtractionConnectionError,
    StructuredExtractionInputError,
    StructuredExtractionProviderError,
    StructuredExtractionRateLimitError,
    StructuredExtractionResponseError,
    extract_structured_data,
)


client = TestClient(app)
structured_extraction_adapter = TypeAdapter(StructuredExtraction)
FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"


def test_backend_root_describes_api_service() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["status"] == "healthy"
    assert response.json()["docs"] == "/docs"


def test_backend_allows_local_frontend_cors_preflight() -> None:
    response = client.options(
        "/api/v1/documents",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8080"
    assert "GET" in response.headers["access-control-allow-methods"]


def test_static_dashboard_frontend_shell_exists() -> None:
    dashboard = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")

    assert "IntelliDoc" in dashboard
    assert 'id="upload-form"' in dashboard
    assert 'name="file"' in dashboard
    assert 'name="document_type"' in dashboard
    assert "/config.js" in dashboard
    assert "/static/css/styles.css" in dashboard
    assert "/static/js/app.js" in dashboard
    assert "{{" not in dashboard


def test_dashboard_contains_exact_supported_document_type_values() -> None:
    dashboard = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
    option_values = [
        value
        for value in re.findall(r'<option value="([^"]*)"', dashboard)
        if value
    ]

    assert option_values == [
        "invoice",
        "balance_sheet",
        "profit_and_loss",
        "cash_flow_statement",
    ]


@pytest.mark.parametrize("path", ["static/css/styles.css", "static/js/app.js"])
def test_frontend_static_assets_exist(path: str) -> None:
    assert (FRONTEND_ROOT / path).read_text(encoding="utf-8").strip()


def test_frontend_javascript_uses_existing_api_contract() -> None:
    javascript = (FRONTEND_ROOT / "static/js/app.js").read_text(encoding="utf-8")

    assert 'formData.append("file"' in javascript
    assert 'formData.append("document_type"' in javascript
    assert 'apiUrl("/api/v1/documents/process")' in javascript
    assert 'apiUrl("/api/v1/documents")' in javascript
    assert "apiUrl(`/api/v1/documents/${encodeURIComponent(documentName)}`)" in javascript
    assert "function documentNameFromPath(" in javascript
    assert "function formatFieldName(" in javascript
    assert "function formatConfidence(" in javascript
    assert "function renderEvidence(" in javascript
    assert "function groupChecksByPeriod(" in javascript
    assert "function renderInputValues(" in javascript
    assert "parseFloat(" not in javascript
    assert ".innerHTML" not in javascript


def test_api_only_backend_openapi_surface() -> None:
    assert set(app.openapi()["paths"]) == {
        "/api/v1/health",
        "/api/v1/documents/process",
        "/api/v1/documents",
        "/api/v1/documents/{document_name}",
    }


def test_static_document_result_shell_contains_no_server_template_values() -> None:
    result_page = (FRONTEND_ROOT / "document_result.html").read_text(
        encoding="utf-8"
    )

    assert 'data-page="result"' in result_page
    assert "{{" not in result_page
    assert "/config.js" in result_page
    assert "/static/js/app.js" in result_page
    for element_id in (
        "header-financial-status",
        "extracted-fields",
        "validation-periods",
        "extracted-text-disclosure",
        "raw-text-disclosure",
        "ocr-details-disclosure",
        "raw-response-json",
    ):
        assert f'id="{element_id}"' in result_page


def test_frontend_config_points_to_separate_backend() -> None:
    config = (FRONTEND_ROOT / "config.js").read_text(encoding="utf-8")

    assert "https://tannu-intellidoc-backend.onrender.com" in config


def test_result_rendering_styles_are_served() -> None:
    stylesheet = (FRONTEND_ROOT / "static/css/styles.css").read_text(
        encoding="utf-8"
    )

    for selector in (
        ".extracted-field-card",
        ".confidence-label",
        ".evidence-item",
        ".validation-check",
        ".input-values",
        ".disclosure",
    ):
        assert selector in stylesheet


def valid_field(*, value: object = "sample value") -> dict[str, Any]:
    return {
        "field": "sample_field",
        "value": value,
        "evidence": [
            {
                "source_text": "Grounding text from the document",
                "page_number": 1,
                "bounding_box": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 100.0,
                    "height": 15.0,
                },
            }
        ],
        "confidence": 0.9,
    }


@pytest.mark.parametrize(
    "document_type",
    [
        "invoice",
        "balance_sheet",
        "profit_and_loss",
        "cash_flow_statement",
    ],
)
def test_process_endpoint_propagates_document_type(
    document_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_process_document(
        file_name: str,
        content: bytes,
        received_document_type: str,
        _session: object,
    ) -> dict[str, object]:
        received.update(
            file_name=file_name,
            content=content,
            document_type=received_document_type,
        )
        return {
            "file_name": file_name,
            "document_type": received_document_type,
            "file_validation": {
                "file_type": "application/pdf",
                "is_supported": True,
                "is_readable": True,
                "page_count": 1,
                "status": "PASS",
            },
            "text_extraction": {
                "extracted_text": "Readable document text",
                "extraction_method": "native_text",
                "page_count": 1,
                "status": "PASS",
            },
            "structured_extraction": {
                "document_type": received_document_type,
                "extracted_fields": [valid_field()],
            },
        }

    monkeypatch.setattr(
        "app.api.routes.documents.process_document_file",
        fake_process_document,
    )

    response = client.post(
        "/api/v1/documents/process",
        files={"file": ("sample.pdf", b"pdf content", "application/pdf")},
        data={"document_type": document_type},
    )

    assert response.status_code == 202
    assert received == {
        "file_name": "sample.pdf",
        "content": b"pdf content",
        "document_type": document_type,
    }
    assert response.json()["document_type"] == document_type
    assert response.json()["structured_extraction"]["document_type"] == document_type


def test_health_remains_responsive_while_document_processing_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processing_started = Event()
    finish_processing = Event()

    def slow_process_document(*_args: object) -> dict[str, object]:
        processing_started.set()
        finish_processing.wait(timeout=5)
        raise document_service.DocumentValidationError(
            {"file_validation": {"status": "FAIL"}}
        )

    monkeypatch.setattr(
        "app.api.routes.documents.process_document_file",
        slow_process_document,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        processing_request = executor.submit(
            client.post,
            "/api/v1/documents/process",
            files={"file": ("scan.pdf", b"pdf content", "application/pdf")},
            data={"document_type": "cash_flow_statement"},
        )
        assert processing_started.wait(timeout=5)
        try:
            health_response = client.get("/api/v1/health")
        finally:
            finish_processing.set()
        processing_response = processing_request.result(timeout=5)

    assert health_response.status_code == 200
    assert processing_response.status_code == 400


@pytest.mark.parametrize(
    ("document_type", "expected_model"),
    [
        ("invoice", InvoiceExtraction),
        ("balance_sheet", BalanceSheetExtraction),
        ("profit_and_loss", ProfitAndLossExtraction),
        ("cash_flow_statement", CashFlowStatementExtraction),
    ],
)
def test_supported_structured_extraction_contracts(
    document_type: str,
    expected_model: type,
) -> None:
    result = structured_extraction_adapter.validate_python(
        {
            "document_type": document_type,
            "extracted_fields": [valid_field()],
        }
    )

    assert isinstance(result, expected_model)
    assert result.document_type == document_type


@pytest.mark.parametrize(
    "malformed_data",
    [
        {
            "document_type": "invoice",
            "extracted_fields": [valid_field()],
            "unexpected": "not allowed",
        },
        {
            "document_type": "invoice",
            "extracted_fields": [
                {**valid_field(), "field": "Invalid Field Name"}
            ],
        },
        {
            "document_type": "invoice",
            "extracted_fields": [
                {**valid_field(), "value": "applicable", "evidence": []}
            ],
        },
        {
            "document_type": "invoice",
            "extracted_fields": [valid_field(), valid_field()],
        },
    ],
)
def test_malformed_structured_extraction_is_rejected(
    malformed_data: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        structured_extraction_adapter.validate_python(malformed_data)


def test_not_applicable_is_preserved_as_an_explicit_value() -> None:
    field = valid_field(value="NOT_APPLICABLE")
    field["evidence"] = []

    result = InvoiceExtraction(
        document_type="invoice",
        extracted_fields=[field],
    )

    assert result.extracted_fields[0].value == "NOT_APPLICABLE"
    assert result.model_dump()["extracted_fields"][0]["value"] == "NOT_APPLICABLE"


def test_not_applicable_cannot_claim_source_evidence() -> None:
    with pytest.raises(ValidationError, match="must not include source evidence"):
        InvoiceExtraction(
            document_type="invoice",
            extracted_fields=[valid_field(value="NOT_APPLICABLE")],
        )


def configure_mock_groq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_content: str | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        extraction_service,
        "settings",
        SimpleNamespace(
            llm_provider="groq",
            llm_model="qwen/qwen3.8-27b",
            llm_api_key="test-only-key",
            llm_max_completion_tokens=4096,
        ),
    )

    class FakeCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            if error is not None:
                raise error
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=response_content)
                    )
                ]
            )

    class FakeGroq:
        def __init__(self, *, api_key: str) -> None:
            captured["client_created"] = True
            captured["api_key_supplied"] = bool(api_key)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(extraction_service, "Groq", FakeGroq)
    return captured


@pytest.mark.parametrize(
    ("document_type", "expected_model", "instruction_fragment"),
    [
        ("invoice", InvoiceExtraction, "line_item_<n>_quantity"),
        (
            "balance_sheet",
            BalanceSheetExtraction,
            "total_capital_and_liabilities",
        ),
        (
            "profit_and_loss",
            ProfitAndLossExtraction,
            "consolidated_net_profit_before_minority_interest",
        ),
        (
            "cash_flow_statement",
            CashFlowStatementExtraction,
            "net_cash_flow_from_operating_activities",
        ),
    ],
)
def test_groq_structured_extraction_for_each_document_type(
    document_type: str,
    expected_model: type,
    instruction_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_result = {
        "document_type": document_type,
        "extracted_fields": [valid_field()],
    }
    captured = configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(provider_result),
    )

    result = extract_structured_data(
        document_type=document_type,
        extracted_text="Readable financial document text",
        pages=[
            {
                "page_number": 1,
                "raw_text": "internal raw OCR text",
                "ocr_details": {"engine": "internal"},
                "lines": [
                    {
                        "text": "Grounding text from the document",
                        "bounding_box": {
                            "x": 10,
                            "y": 20,
                            "width": 100,
                            "height": 15,
                        },
                    }
                ],
            }
        ],
    )

    assert isinstance(result, expected_model)
    assert result.document_type == document_type
    assert captured["model"] == "qwen/qwen3.8-27b"
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == (
        f"{document_type}_extraction"
    )
    provider_schema = response_format["json_schema"]["schema"]
    evidence_schema = provider_schema["$defs"]["ExtractionEvidence"]
    value_schema = provider_schema["$defs"]["ExtractedField"]["properties"][
        "value"
    ]
    assert value_schema == {"title": "Value", "type": "string"}
    assert set(evidence_schema["required"]) == {
        "source_text",
        "page_number",
        "bounding_box",
    }
    assert "default" not in evidence_schema["properties"]["page_number"]
    assert captured["temperature"] == 0
    assert captured["max_completion_tokens"] == 4096
    assert captured["reasoning_effort"] == "none"
    assert captured["api_key_supplied"] is True
    user_prompt = captured["messages"][1]["content"]
    assert document_type in user_prompt
    assert instruction_fragment in user_prompt
    assert "bounding_box" in user_prompt
    assert "The response must match this JSON Schema exactly" not in user_prompt
    assert "internal raw OCR text" not in user_prompt
    assert "ocr_details" not in user_prompt
    assert user_prompt.count("Readable financial document text") == 0

    system_prompt = captured["messages"][0]["content"]
    normalized_system_prompt = " ".join(system_prompt.split())
    assert "Never return null, an array" in system_prompt
    assert "as an exact string copied from" in system_prompt
    assert "a JSON number, or a JSON boolean" in normalized_system_prompt
    assert "exactly one concise, verbatim evidence entry" in (
        normalized_system_prompt
    )
    assert "evidence must literally contain the exact returned value" in (
        normalized_system_prompt
    )
    assert "Never infer a missing value from arithmetic" in (
        normalized_system_prompt
    )

    if document_type == "invoice":
        assert "line_item_<n>_quantity" in user_prompt
        assert "Never place line items in an array" in user_prompt
        assert "Do not derive a missing quantity" in user_prompt


def test_groq_not_applicable_value_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = valid_field(value="NOT_APPLICABLE")
    field["evidence"] = []
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [field],
            }
        ),
    )

    result = extract_structured_data(
        document_type="invoice",
        extracted_text="Readable financial document text",
    )

    assert result.extracted_fields[0].value == "NOT_APPLICABLE"


@pytest.mark.parametrize(
    "response_content",
    [
        "not valid JSON",
        json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [
                    {**valid_field(), "confidence": 1.5}
                ],
            }
        ),
        json.dumps(
            {
                "document_type": "balance_sheet",
                "extracted_fields": [valid_field()],
            }
        ),
    ],
)
def test_malformed_groq_output_is_rejected_without_repair(
    response_content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_mock_groq(monkeypatch, response_content=response_content)

    with pytest.raises(StructuredExtractionResponseError):
        extract_structured_data(
            document_type="invoice",
            extracted_text="Readable financial document text",
        )


@pytest.mark.parametrize(
    "field_update",
    [
        {
            "evidence": [
                {
                    "source_text": "Text not present in the source",
                    "page_number": 1,
                }
            ]
        },
        {
            "evidence": [
                {
                    "source_text": "Grounding text from the document",
                    "page_number": 2,
                }
            ]
        },
        {
            "evidence": [
                {
                    "source_text": "Grounding text from the document",
                    "page_number": 1,
                    "bounding_box": {
                        "x": 999.0,
                        "y": 20.0,
                        "width": 100.0,
                        "height": 15.0,
                    },
                }
            ]
        },
    ],
)
def test_ungrounded_provider_evidence_is_rejected(
    field_update: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [{**valid_field(), **field_update}],
            }
        ),
    )

    with pytest.raises(StructuredExtractionResponseError, match="not grounded"):
        extract_structured_data(
            document_type="invoice",
            extracted_text="Grounding text from the document",
            pages=[
                {
                    "page_number": 1,
                    "lines": [
                        {
                            "text": "Grounding text from the document",
                            "bounding_box": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 100.0,
                                "height": 15.0,
                            },
                        }
                    ],
                }
            ],
        )


def test_page_and_bounding_box_are_added_from_unique_ocr_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = valid_field()
    field["evidence"] = [
        {"source_text": "Grounding text from the document"}
    ]
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [field],
            }
        ),
    )

    result = extract_structured_data(
        document_type="invoice",
        extracted_text="Grounding text from the document",
        pages=[
            {
                "page_number": 1,
                "lines": [
                    {
                        "text": "Grounding text from the document",
                        "bounding_box": {
                            "x": 10,
                            "y": 20,
                            "width": 100,
                            "height": 15,
                        },
                    }
                ],
            }
        ],
    )

    evidence = result.extracted_fields[0].evidence[0]
    assert evidence.page_number == 1
    assert evidence.bounding_box is not None
    assert evidence.bounding_box.model_dump() == {
        "x": 10.0,
        "y": 20.0,
        "width": 100.0,
        "height": 15.0,
    }


def test_evidence_grounding_tolerates_ocr_whitespace_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = valid_field(value="8,923,441,607")
    field["evidence"] = [
        {
            "source_text": "Total Assets 8,923,441,607",
            "page_number": 1,
        }
    ]
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "balance_sheet",
                "extracted_fields": [field],
            }
        ),
    )

    result = extract_structured_data(
        document_type="balance_sheet",
        extracted_text="Total Assets     8,923,441,607",
        pages=[
            {
                "page_number": 1,
                "text": "Total Assets     8,923,441,607",
            }
        ],
    )

    assert result.extracted_fields[0].value == "8,923,441,607"


@pytest.mark.parametrize(
    "financial_value",
    [
        "$1,234.50",
        "€8,923,441,607",
        "£-300.25",
        "₹2,000",
        "(2,000)",
        "8%",
        "Nov 03, 2022",
        "31-Mar-2024",
    ],
)
def test_financial_value_formatting_is_preserved_from_evidence(
    financial_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_line = f"Reported amount {financial_value}"
    field = valid_field(value=financial_value)
    field["evidence"] = [{"source_text": source_line, "page_number": 1}]
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "balance_sheet",
                "extracted_fields": [field],
            }
        ),
    )

    result = extract_structured_data(
        document_type="balance_sheet",
        extracted_text=source_line,
        pages=[{"page_number": 1, "text": source_line}],
    )

    assert result.extracted_fields[0].value == financial_value


@pytest.mark.parametrize(
    ("returned_value", "source_value"),
    [
        ("1234.50", "$1,234.50"),
        ("157.48", "$157.48"),
        ("8", "8%"),
        ("2022-11-03", "Nov 03, 2022"),
    ],
)
def test_financial_or_date_value_changed_from_evidence_is_rejected(
    returned_value: str,
    source_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = valid_field(value=returned_value)
    field["evidence"] = [
        {"source_text": f"Reported value {source_value}", "page_number": 1}
    ]
    configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [field],
            }
        ),
    )

    with pytest.raises(
        StructuredExtractionResponseError,
        match="ungrounded.*field 'sample_field'",
    ):
        extract_structured_data(
            document_type="invoice",
            extracted_text=f"Reported value {source_value}",
            pages=[
                {"page_number": 1, "text": f"Reported value {source_value}"}
            ],
        )


def test_missing_field_is_not_added_as_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [valid_field()],
            }
        ),
    )

    result = extract_structured_data(
        document_type="invoice",
        extracted_text="Grounding text from the document",
        pages=[
            {
                "page_number": 1,
                "lines": [
                    {
                        "text": "Grounding text from the document",
                        "bounding_box": {
                            "x": 10,
                            "y": 20,
                            "width": 100,
                            "height": 15,
                        },
                    }
                ],
            }
        ],
    )

    assert [field.field for field in result.extracted_fields] == ["sample_field"]
    system_prompt = captured["messages"][0]["content"]
    assert "missing, unreadable, or uncertain, omit it" in system_prompt
    assert "never substitute" in system_prompt


def test_missing_llm_api_key_is_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extraction_service,
        "settings",
        SimpleNamespace(
            llm_provider="groq",
            llm_model="qwen/qwen3.8-27b",
            llm_api_key="",
            llm_max_completion_tokens=4096,
        ),
    )

    with pytest.raises(
        StructuredExtractionConfigurationError,
        match="LLM_API_KEY is not configured",
    ):
        extract_structured_data(
            document_type="invoice",
            extracted_text="Readable financial document text",
        )


@pytest.mark.parametrize(
    ("provider", "model", "max_completion_tokens"),
    [
        ("unsupported", "qwen/qwen3.8-27b", 4096),
        ("groq", "", 4096),
        ("groq", "qwen/qwen3.6-27b", 4096),
        ("groq", "qwen/qwen3.8-27b", 0),
        ("groq", "qwen/qwen3.8-27b", True),
    ],
)
def test_invalid_llm_configuration_is_controlled(
    provider: str,
    model: str,
    max_completion_tokens: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extraction_service,
        "settings",
        SimpleNamespace(
            llm_provider=provider,
            llm_model=model,
            llm_api_key="test-only-key",
            llm_max_completion_tokens=max_completion_tokens,
        ),
    )

    with pytest.raises(StructuredExtractionConfigurationError):
        extract_structured_data(
            document_type="invoice",
            extracted_text="Readable financial document text",
        )


def groq_status_error(error_type: type, status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.groq.com/v1/test")
    response = httpx.Response(status_code, request=request)
    return error_type("Mock provider failure", response=response, body=None)


@pytest.mark.parametrize(
    ("provider_error", "application_error"),
    [
        (groq_status_error(AuthenticationError, 401), StructuredExtractionAuthenticationError),
        (groq_status_error(RateLimitError, 429), StructuredExtractionRateLimitError),
        (groq_status_error(InternalServerError, 500), StructuredExtractionProviderError),
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://api.groq.com/v1/test")
            ),
            StructuredExtractionConnectionError,
        ),
        (RuntimeError("unexpected SDK failure"), StructuredExtractionProviderError),
    ],
)
def test_groq_failures_are_converted_to_controlled_errors(
    provider_error: Exception,
    application_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_mock_groq(monkeypatch, error=provider_error)

    with pytest.raises(application_error) as captured:
        extract_structured_data(
            document_type="invoice",
            extracted_text="Readable financial document text",
        )

    assert "test-only-key" not in str(captured.value)
    assert "Mock provider failure" not in str(captured.value)


def test_oversized_extracted_text_is_rejected_before_groq_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = configure_mock_groq(
        monkeypatch,
        response_content=json.dumps(
            {
                "document_type": "invoice",
                "extracted_fields": [valid_field()],
            }
        ),
    )

    with pytest.raises(StructuredExtractionInputError, match="too large"):
        extract_structured_data(
            document_type="invoice",
            extracted_text="x" * (MAX_EXTRACTED_TEXT_CHARACTERS + 1),
        )

    assert "client_created" not in captured


def test_document_service_passes_text_and_pages_to_structured_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}
    financial_received: list[object] = []
    pages = [{"page_number": 1, "text": "Extracted page text"}]
    structured_result = InvoiceExtraction(
        document_type="invoice",
        extracted_fields=[valid_field()],
    )
    monkeypatch.setattr(
        document_service,
        "validate_document",
        lambda _file_name, _content: {
            "file_name": "sample.pdf",
            "file_validation": {
                "file_type": "application/pdf",
                "is_supported": True,
                "is_readable": True,
                "page_count": 1,
                "status": "PASS",
            },
        },
    )
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda _file_name, _content: {
            "extracted_text": "Extracted page text",
            "extraction_method": "native_text",
            "page_count": 1,
            "status": "PASS",
            "pages": pages,
        },
    )

    def fake_structured_extraction(
        *,
        document_type: str,
        extracted_text: str,
        pages: list[dict[str, object]],
    ) -> InvoiceExtraction:
        received.update(
            document_type=document_type,
            extracted_text=extracted_text,
            pages=pages,
        )
        return structured_result

    monkeypatch.setattr(
        document_service,
        "extract_structured_data",
        fake_structured_extraction,
    )
    financial_result = object()
    monkeypatch.setattr(
        document_service,
        "validate_financial_data",
        lambda extraction: financial_received.append(extraction) or financial_result,
    )
    persisted: list[tuple[object, dict[str, object]]] = []
    session = object()
    monkeypatch.setattr(
        document_service,
        "create_document_result",
        lambda received_session, processing_result: persisted.append(
            (received_session, processing_result)
        ),
    )

    result = document_service.process_document(
        "sample.pdf",
        b"pdf content",
        "invoice",
        session,
    )

    assert received == {
        "document_type": "invoice",
        "extracted_text": "Extracted page text",
        "pages": pages,
    }
    assert result["structured_extraction"] is structured_result
    assert financial_received == [structured_result]
    assert result["financial_validation"] is financial_result
    assert persisted == [(session, result)]


def test_evidence_accepts_optional_page_and_ocr_coordinates() -> None:
    result = InvoiceExtraction(
        document_type="invoice",
        extracted_fields=[valid_field()],
    )

    evidence = result.extracted_fields[0].evidence[0]
    assert evidence.source_text == "Grounding text from the document"
    assert evidence.page_number == 1
    assert evidence.bounding_box is not None
    assert evidence.bounding_box.x == 10.0


@pytest.mark.parametrize("confidence", [-0.01, 1.01, "0.9"])
def test_confidence_outside_strict_zero_to_one_range_is_rejected(
    confidence: object,
) -> None:
    field = valid_field()
    field["confidence"] = confidence

    with pytest.raises(ValidationError):
        InvoiceExtraction(
            document_type="invoice",
            extracted_fields=[field],
        )


@pytest.mark.parametrize(
    "invalid_evidence",
    [
        {"source_text": "", "page_number": 1},
        {"source_text": "valid text", "page_number": 0},
        {
            "source_text": "valid text",
            "bounding_box": {"x": -1.0, "y": 0.0, "width": 5.0, "height": 5.0},
        },
    ],
)
def test_malformed_evidence_is_rejected(
    invalid_evidence: dict[str, object],
) -> None:
    field = valid_field()
    field["evidence"] = [invalid_evidence]

    with pytest.raises(ValidationError):
        InvoiceExtraction(
            document_type="invoice",
            extracted_fields=[field],
        )
