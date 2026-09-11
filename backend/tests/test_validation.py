from io import BytesIO
from decimal import Decimal
from typing import Any

import pytest
from PIL import Image
from pypdf import PdfWriter

from app.schemas.extraction import (
    BalanceSheetExtraction,
    CashFlowStatementExtraction,
    InvoiceExtraction,
    ProfitAndLossExtraction,
)
from app.services.document_validation_service import validate_document
from app.services.financial_validation_service import (
    parse_financial_value,
    validate_financial_data,
)


def make_pdf(page_count: int) -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def make_image(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(output, format=image_format)
    return output.getvalue()


def test_valid_pdf_with_at_most_three_pages() -> None:
    result = validate_document("invoice.pdf", make_pdf(3))

    assert result == {
        "file_name": "invoice.pdf",
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 3,
            "status": "PASS",
        },
    }


def test_pdf_with_more_than_three_pages_is_rejected() -> None:
    result = validate_document("long-report.pdf", make_pdf(4))

    validation = result["file_validation"]
    assert validation["is_readable"] is True
    assert validation["page_count"] == 4
    assert validation["status"] == "FAIL"


def test_unsupported_file_type_is_rejected() -> None:
    result = validate_document("notes.txt", b"plain text")

    validation = result["file_validation"]
    assert validation["is_supported"] is False
    assert validation["is_readable"] is False
    assert validation["status"] == "FAIL"


def test_empty_file_is_rejected() -> None:
    result = validate_document("empty.pdf", b"")

    validation = result["file_validation"]
    assert validation["is_supported"] is True
    assert validation["is_readable"] is False
    assert validation["status"] == "FAIL"


def test_corrupted_pdf_is_rejected() -> None:
    result = validate_document("corrupted.pdf", b"%PDF-1.7\nnot a valid PDF")

    validation = result["file_validation"]
    assert validation["is_supported"] is True
    assert validation["is_readable"] is False
    assert validation["page_count"] is None
    assert validation["status"] == "FAIL"


@pytest.mark.parametrize(
    ("file_name", "image_format", "expected_type"),
    [
        ("scan.png", "PNG", "image/png"),
        ("scan.jpg", "JPEG", "image/jpeg"),
        ("scan.jpeg", "JPEG", "image/jpeg"),
    ],
)
def test_valid_png_and_jpeg_inputs(
    file_name: str,
    image_format: str,
    expected_type: str,
) -> None:
    result = validate_document(file_name, make_image(image_format))

    assert result["file_validation"] == {
        "file_type": expected_type,
        "is_supported": True,
        "is_readable": True,
        "page_count": None,
        "status": "PASS",
    }


def extracted_field(name: str, value: object) -> dict[str, Any]:
    return {
        "field": name,
        "value": value,
        "evidence": (
            []
            if value == "NOT_APPLICABLE"
            else [{"source_text": f"{name}: {value}"}]
        ),
        "confidence": 0.9,
    }


def financial_extraction(
    document_type: str,
    values: dict[str, object],
) -> object:
    model_by_type = {
        "invoice": InvoiceExtraction,
        "balance_sheet": BalanceSheetExtraction,
        "profit_and_loss": ProfitAndLossExtraction,
        "cash_flow_statement": CashFlowStatementExtraction,
    }
    return model_by_type[document_type](
        document_type=document_type,
        extracted_fields=[
            extracted_field(name, value) for name, value in values.items()
        ],
    )


def checks_by_name(result: object, rule_name: str) -> list[object]:
    return [check for check in result.checks if check.rule_name == rule_name]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("8,923,441,607", Decimal("8923441607")),
        ("8,923,441.50", Decimal("8923441.50")),
        ("₹8,923,441", Decimal("8923441")),
        ("$1,234.50", Decimal("1234.50")),
        ("-2,000", Decimal("-2000")),
        ("(2,000)", Decimal("-2000")),
        ("[2,000]", Decimal("-2000")),
    ],
)
def test_financial_numeric_parser(
    raw_value: str,
    expected: Decimal,
) -> None:
    assert parse_financial_value(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value",
    ["", "1,23,4", "1 000", "--10", "(10", "ten", "10%", "-"],
)
def test_financial_numeric_parser_rejects_malformed_values(raw_value: str) -> None:
    with pytest.raises(ValueError):
        parse_financial_value(raw_value)


@pytest.mark.parametrize(
    ("line_total", "expected_status"),
    [("20.00", "PASS"), ("20.02", "FAIL")],
)
def test_invoice_quantity_times_unit_price(
    line_total: str,
    expected_status: str,
) -> None:
    extraction = financial_extraction(
        "invoice",
        {
            "line_item_1_quantity": "2",
            "line_item_1_unit_price": "10.00",
            "line_item_1_line_total": line_total,
            "subtotal": line_total,
        },
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(result, "invoice_line_item_1")[0].status == expected_status


@pytest.mark.parametrize(
    ("subtotal", "expected_status"),
    [("50.00", "PASS"), ("51.00", "FAIL")],
)
def test_invoice_line_totals_reconciliation(
    subtotal: str,
    expected_status: str,
) -> None:
    extraction = financial_extraction(
        "invoice",
        {
            "line_item_1_quantity": "2",
            "line_item_1_unit_price": "10",
            "line_item_1_line_total": "20",
            "line_item_2_quantity": "3",
            "line_item_2_unit_price": "10",
            "line_item_2_line_total": "30",
            "subtotal": subtotal,
        },
    )

    result = validate_financial_data(extraction)

    check = checks_by_name(result, "invoice_line_totals_reconciliation")[0]
    assert check.status == expected_status


@pytest.mark.parametrize(
    ("total", "expected_status"),
    [("108", "PASS"), ("109", "FAIL")],
)
def test_invoice_tax_reconciliation(total: str, expected_status: str) -> None:
    extraction = financial_extraction(
        "invoice",
        {"taxable_amount": "100", "tax_amount": "8", "total_amount": total},
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(result, "invoice_tax_reconciliation")[0].status == expected_status


def test_invoice_cash_paid_minus_total_equals_change() -> None:
    extraction = financial_extraction(
        "invoice",
        {"cash_paid": "200", "total_amount": "157.48", "change": "42.52"},
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(result, "invoice_cash_change_reconciliation")[0].status == "PASS"


def test_invoice_tax_inclusive_total_is_not_double_taxed() -> None:
    extraction = financial_extraction(
        "invoice",
        {
            "line_item_1_quantity": "1",
            "line_item_1_unit_price": "108",
            "line_item_1_line_total": "108",
            "total_amount": "108",
            "tax_amount": "8",
            "tax_inclusive": True,
        },
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(result, "invoice_tax_reconciliation")[0].status == "NOT_APPLICABLE"
    line_total_check = checks_by_name(
        result,
        "invoice_line_totals_reconciliation",
    )[0]
    assert line_total_check.status == "PASS"
    assert line_total_check.formula == "sum(line_totals) = tax_inclusive_total"


def test_invoice_missing_fields_are_not_applicable() -> None:
    extraction = financial_extraction("invoice", {"invoice_number": "825"})

    result = validate_financial_data(extraction)

    assert result.status == "NOT_APPLICABLE"
    assert all(check.status == "NOT_APPLICABLE" for check in result.checks)


@pytest.mark.parametrize(
    ("assets", "expected_status"),
    [("1000", "PASS"), ("999", "FAIL")],
)
def test_balance_sheet_main_equality(assets: str, expected_status: str) -> None:
    extraction = financial_extraction(
        "balance_sheet",
        {
            "total_capital_and_liabilities": "1,000",
            "total_assets": assets,
        },
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(result, "balance_sheet_main_equality")[0].status == expected_status


@pytest.mark.parametrize(
    ("reported_total", "expected_status"),
    [("1000", "PASS"), ("1001", "FAIL")],
)
def test_balance_sheet_component_reconciliation(
    reported_total: str,
    expected_status: str,
) -> None:
    extraction = financial_extraction(
        "balance_sheet",
        {
            "capital_and_liability_component_capital": "100",
            "capital_and_liability_component_reserves": "200",
            "capital_and_liability_component_deposits": "700",
            "total_capital_and_liabilities": reported_total,
            "asset_component_cash": "100",
            "asset_component_investments": "400",
            "asset_component_advances": "500",
            "total_assets": reported_total,
        },
    )

    result = validate_financial_data(extraction)

    assert checks_by_name(
        result,
        "balance_sheet_capital_liability_components",
    )[0].status == expected_status
    assert checks_by_name(
        result,
        "balance_sheet_asset_components",
    )[0].status == expected_status


def test_balance_sheet_comparative_periods_remain_independent() -> None:
    extraction = financial_extraction(
        "balance_sheet",
        {
            "total_capital_and_liabilities_2025": "100",
            "total_assets_2025": "100",
            "total_capital_and_liabilities_2024": "90",
            "total_assets_2024": "89",
        },
    )

    result = validate_financial_data(extraction)
    checks = checks_by_name(result, "balance_sheet_main_equality")

    assert {(check.period, check.status) for check in checks} == {
        ("2025", "PASS"),
        ("2024", "FAIL"),
    }


def test_balance_sheet_missing_total_is_not_applicable() -> None:
    extraction = financial_extraction("balance_sheet", {"total_assets": "100"})

    result = validate_financial_data(extraction)

    check = checks_by_name(result, "balance_sheet_main_equality")[0]
    assert check.status == "NOT_APPLICABLE"
    assert "total_capital_and_liabilities (missing)" in check.unavailable_inputs


def profit_and_loss_values() -> dict[str, object]:
    return {
        "interest_earned": "100",
        "other_income": "20",
        "total_income": "120",
        "interest_expended": "40",
        "operating_expenses": "50",
        "provisions_and_contingencies": "10",
        "total_expenditure": "100",
        "consolidated_net_profit_before_minority_interest": "20",
        "minority_interest": "5",
        "consolidated_net_profit_attributable_to_the_group": "15",
        "current_profit": "15",
        "brought_forward_profit": "5",
        "total_available_for_appropriation": "20",
    }


@pytest.mark.parametrize(
    "rule_name",
    [
        "profit_and_loss_income_reconciliation",
        "profit_and_loss_expenditure_reconciliation",
        "profit_and_loss_net_profit_before_minority",
        "profit_and_loss_minority_interest_reconciliation",
        "profit_and_loss_appropriation_reconciliation",
    ],
)
def test_profit_and_loss_reconciliations_pass(rule_name: str) -> None:
    extraction = financial_extraction("profit_and_loss", profit_and_loss_values())

    result = validate_financial_data(extraction)

    assert checks_by_name(result, rule_name)[0].status == "PASS"


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "rule_name"),
    [
        ("total_income", "121", "profit_and_loss_income_reconciliation"),
        (
            "total_expenditure",
            "101",
            "profit_and_loss_expenditure_reconciliation",
        ),
        (
            "consolidated_net_profit_before_minority_interest",
            "21",
            "profit_and_loss_net_profit_before_minority",
        ),
        (
            "consolidated_net_profit_attributable_to_the_group",
            "16",
            "profit_and_loss_minority_interest_reconciliation",
        ),
        (
            "total_available_for_appropriation",
            "21",
            "profit_and_loss_appropriation_reconciliation",
        ),
    ],
)
def test_profit_and_loss_reconciliation_failures(
    changed_field: str,
    changed_value: str,
    rule_name: str,
) -> None:
    values = profit_and_loss_values()
    values[changed_field] = changed_value
    extraction = financial_extraction("profit_and_loss", values)

    result = validate_financial_data(extraction)

    assert checks_by_name(result, rule_name)[0].status == "FAIL"


def test_profit_and_loss_comparative_periods_remain_independent() -> None:
    extraction = financial_extraction(
        "profit_and_loss",
        {
            "interest_earned_2025": "100",
            "other_income_2025": "20",
            "total_income_2025": "120",
            "interest_earned_2024": "90",
            "other_income_2024": "10",
            "total_income_2024": "101",
        },
    )

    result = validate_financial_data(extraction)
    checks = checks_by_name(result, "profit_and_loss_income_reconciliation")

    assert {(check.period, check.status) for check in checks} == {
        ("2025", "PASS"),
        ("2024", "FAIL"),
    }


def test_profit_and_loss_missing_values_are_not_applicable() -> None:
    extraction = financial_extraction("profit_and_loss", {"total_income": "100"})

    result = validate_financial_data(extraction)

    assert checks_by_name(
        result,
        "profit_and_loss_income_reconciliation",
    )[0].status == "NOT_APPLICABLE"


def cash_flow_values() -> dict[str, object]:
    return {
        "net_cash_flow_from_operating_activities": "100",
        "net_cash_flow_from_investing_activities": "(20)",
        "net_cash_flow_from_financing_activities": "[30]",
        "fx_translation_adjustment": "0",
        "net_increase_in_cash_and_cash_equivalents": "50",
        "opening_cash_and_cash_equivalents": "1,000",
        "cash_acquired_on_amalgamation": "NOT_APPLICABLE",
        "closing_cash_and_cash_equivalents": "1,050",
    }


def test_cash_flow_reconciliations_and_parenthesized_negatives_pass() -> None:
    extraction = financial_extraction("cash_flow_statement", cash_flow_values())

    result = validate_financial_data(extraction)

    net_check = checks_by_name(result, "cash_flow_net_increase_reconciliation")[0]
    closing_check = checks_by_name(
        result,
        "cash_flow_opening_closing_reconciliation",
    )[0]
    assert net_check.status == "PASS"
    assert net_check.input_values["investing_cash_flow"] == Decimal("-20")
    assert net_check.input_values["financing_cash_flow"] == Decimal("-30")
    assert closing_check.status == "PASS"


def test_cash_flow_closing_cash_mismatch_fails() -> None:
    values = cash_flow_values()
    values["closing_cash_and_cash_equivalents"] = "1,051"
    extraction = financial_extraction("cash_flow_statement", values)

    result = validate_financial_data(extraction)

    assert checks_by_name(
        result,
        "cash_flow_opening_closing_reconciliation",
    )[0].status == "FAIL"


def test_cash_flow_comparative_periods_remain_independent() -> None:
    extraction = financial_extraction(
        "cash_flow_statement",
        {
            "net_cash_flow_from_operating_activities_2025": "100",
            "net_cash_flow_from_investing_activities_2025": "-20",
            "net_cash_flow_from_financing_activities_2025": "-30",
            "fx_translation_adjustment_2025": "0",
            "net_increase_in_cash_and_cash_equivalents_2025": "50",
            "net_cash_flow_from_operating_activities_2024": "100",
            "net_cash_flow_from_investing_activities_2024": "-20",
            "net_cash_flow_from_financing_activities_2024": "-30",
            "fx_translation_adjustment_2024": "0",
            "net_increase_in_cash_and_cash_equivalents_2024": "49",
        },
    )

    result = validate_financial_data(extraction)
    checks = checks_by_name(result, "cash_flow_net_increase_reconciliation")

    assert {(check.period, check.status) for check in checks} == {
        ("2025", "PASS"),
        ("2024", "FAIL"),
    }


def test_cash_flow_missing_values_are_not_applicable() -> None:
    extraction = financial_extraction(
        "cash_flow_statement",
        {"opening_cash_and_cash_equivalents": "100"},
    )

    result = validate_financial_data(extraction)

    assert result.status == "NOT_APPLICABLE"


@pytest.mark.parametrize(
    ("reported_total", "expected_status"),
    [("105.01", "PASS"), ("105.011", "FAIL")],
)
def test_financial_tolerance_boundary(
    reported_total: str,
    expected_status: str,
) -> None:
    extraction = financial_extraction(
        "invoice",
        {
            "taxable_amount": "100",
            "tax_amount": "5",
            "total_amount": reported_total,
        },
    )

    result = validate_financial_data(extraction, tolerance="0.01")
    check = checks_by_name(result, "invoice_tax_reconciliation")[0]

    assert check.status == expected_status
    assert check.variance == abs(Decimal("105") - Decimal(reported_total))


def test_malformed_extracted_number_does_not_crash_validation() -> None:
    extraction = financial_extraction(
        "invoice",
        {"taxable_amount": "one hundred", "tax_amount": "5", "total_amount": "105"},
    )

    result = validate_financial_data(extraction)
    check = checks_by_name(result, "invoice_tax_reconciliation")[0]

    assert check.status == "NOT_APPLICABLE"
    assert "taxable_amount (invalid)" in check.unavailable_inputs


def test_financial_validation_result_is_json_compatible() -> None:
    extraction = financial_extraction(
        "balance_sheet",
        {"total_capital_and_liabilities": "100", "total_assets": "100"},
    )

    payload = validate_financial_data(extraction).model_dump(mode="json")

    assert payload["tolerance"] == "0.01"
    assert payload["checks"][0]["calculated_value"] == "100"
