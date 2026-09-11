from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


DocumentType = Literal[
    "invoice",
    "balance_sheet",
    "profit_and_loss",
    "cash_flow_statement",
]
NotApplicable = Literal["NOT_APPLICABLE"]

Coordinate = Annotated[float, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1)]
FieldName = Annotated[
    StrictStr,
    Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$"),
]
ExtractedValue = Union[
    NotApplicable,
    StrictStr,
    StrictInt,
    StrictFloat,
    StrictBool,
    date,
]


class StrictSchemaModel(BaseModel):
    """Base configuration for data received from a future extraction engine."""

    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceBoundingBox(StrictSchemaModel):
    """A source rectangle in the coordinate system emitted by OCR."""

    x: Coordinate
    y: Coordinate
    width: Coordinate
    height: Coordinate


class ExtractionEvidence(StrictSchemaModel):
    """Grounding for one extracted value."""

    source_text: Annotated[StrictStr, Field(min_length=1)]
    page_number: Annotated[int, Field(ge=1)] | None = None
    bounding_box: EvidenceBoundingBox | None = None

    @model_validator(mode="after")
    def reject_blank_source_text(self) -> ExtractionEvidence:
        if not self.source_text.strip():
            raise ValueError("Evidence source_text must not be blank.")
        return self


class ExtractedField(StrictSchemaModel):
    """One grounded value in a document-specific extraction result."""

    field: FieldName
    value: ExtractedValue
    evidence: list[ExtractionEvidence] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_value_and_evidence(self) -> ExtractedField:
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("An extracted value must not be blank.")
        if self.value != "NOT_APPLICABLE" and not self.evidence:
            raise ValueError("Evidence is required for an applicable value.")
        if self.value == "NOT_APPLICABLE" and self.evidence:
            raise ValueError("NOT_APPLICABLE must not include source evidence.")
        return self


class DocumentStructuredExtraction(StrictSchemaModel):
    """Shared contract for every supported document type."""

    extracted_fields: list[ExtractedField] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_field_names(self) -> DocumentStructuredExtraction:
        names = [item.field for item in self.extracted_fields]
        if len(names) != len(set(names)):
            raise ValueError("Each extracted field name must be unique.")
        return self


class InvoiceExtraction(DocumentStructuredExtraction):
    document_type: Literal["invoice"]


class BalanceSheetExtraction(DocumentStructuredExtraction):
    document_type: Literal["balance_sheet"]


class ProfitAndLossExtraction(DocumentStructuredExtraction):
    document_type: Literal["profit_and_loss"]


class CashFlowStatementExtraction(DocumentStructuredExtraction):
    document_type: Literal["cash_flow_statement"]


StructuredExtraction = Annotated[
    Union[
        InvoiceExtraction,
        BalanceSheetExtraction,
        ProfitAndLossExtraction,
        CashFlowStatementExtraction,
    ],
    Field(discriminator="document_type"),
]


FinancialValidationStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE"]
FinancialAmount = Annotated[Decimal, Field(allow_inf_nan=False)]


class FinancialValidationCheck(StrictSchemaModel):
    """One deterministic financial reconciliation performed by Python."""

    rule_name: Annotated[StrictStr, Field(min_length=1)]
    period: Annotated[StrictStr, Field(min_length=1)] | None = None
    formula: Annotated[StrictStr, Field(min_length=1)]
    input_values: dict[StrictStr, FinancialAmount] = Field(default_factory=dict)
    calculated_value: FinancialAmount | None = None
    reported_value: FinancialAmount | None = None
    variance: FinancialAmount | None = None
    status: FinancialValidationStatus
    unavailable_inputs: list[Annotated[StrictStr, Field(min_length=1)]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_status_payload(self) -> FinancialValidationCheck:
        calculated_fields = (
            self.calculated_value,
            self.reported_value,
            self.variance,
        )
        if self.status == "NOT_APPLICABLE":
            if any(value is not None for value in calculated_fields):
                raise ValueError(
                    "NOT_APPLICABLE checks must not contain calculated results."
                )
        elif any(value is None for value in calculated_fields):
            raise ValueError("PASS and FAIL checks require calculated results.")

        if self.variance is not None and self.variance < 0:
            raise ValueError("Validation variance must not be negative.")
        return self


class FinancialValidationResult(StrictSchemaModel):
    """Document-level result for deterministic financial checks."""

    document_type: DocumentType
    tolerance: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    status: FinancialValidationStatus
    checks: list[FinancialValidationCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate_status(self) -> FinancialValidationResult:
        expected_status: FinancialValidationStatus
        if any(check.status == "FAIL" for check in self.checks):
            expected_status = "FAIL"
        elif any(check.status == "PASS" for check in self.checks):
            expected_status = "PASS"
        else:
            expected_status = "NOT_APPLICABLE"

        if self.status != expected_status:
            raise ValueError(
                "Financial validation status must match its check statuses."
            )
        return self
