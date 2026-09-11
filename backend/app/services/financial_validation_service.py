from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable, Literal, Sequence

from app.core.config import settings
from app.schemas.extraction import (
    DocumentStructuredExtraction,
    ExtractedField,
    FinancialValidationCheck,
    FinancialValidationResult,
)


_CURRENCY_SYMBOLS = "₹$€£"
_NUMBER_PATTERN = re.compile(
    r"^(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$"
)
_PERIOD_WORDS = {
    "current_period",
    "previous_period",
    "prior_period",
    "comparative_period",
    "current_year",
    "previous_year",
    "prior_year",
}

FieldState = Literal[
    "available",
    "missing",
    "not_applicable",
    "invalid",
]


@dataclass(frozen=True)
class _ResolvedValue:
    state: FieldState
    value: Decimal | None = None
    source_field: str | None = None


@dataclass(frozen=True)
class _Operand:
    name: str
    aliases: tuple[str, ...]
    coefficient: Decimal = Decimal("1")
    optional_when_explicitly_not_applicable: bool = False


class _FieldAccessor:
    """Resolve only explicitly listed aliases; no fuzzy semantic matching."""

    def __init__(self, extraction: DocumentStructuredExtraction) -> None:
        self._fields = extraction.extracted_fields

    def periods_for(self, aliases: Sequence[str]) -> set[str | None]:
        periods: set[str | None] = set()
        for item in self._fields:
            for alias in aliases:
                matched, period = _match_alias_and_period(item.field, alias)
                if matched:
                    periods.add(period)
                    break
        return periods

    def resolve(
        self,
        aliases: Sequence[str],
        period: str | None,
    ) -> _ResolvedValue:
        # Alias order is intentional and deterministic. More specific canonical
        # names are listed before accepted alternatives in the constants below.
        for alias in aliases:
            for item in self._fields:
                matched, item_period = _match_alias_and_period(item.field, alias)
                if matched and item_period == period:
                    return _resolve_extracted_field(item)
        return _ResolvedValue(state="missing")

    @property
    def fields(self) -> list[ExtractedField]:
        return self._fields


INVOICE_SUBTOTAL = (
    "subtotal",
    "invoice_subtotal",
    "taxable_amount",
)
INVOICE_TAXABLE_AMOUNT = (
    "taxable_amount",
    "subtotal_before_tax",
    "pre_tax_total",
    "subtotal",
    "invoice_subtotal",
)
INVOICE_TAX = ("tax_amount", "sales_tax", "gst_amount", "vat_amount")
INVOICE_TOTAL = (
    "total_amount",
    "total_due",
    "invoice_total",
    "grand_total",
    "total",
)
INVOICE_CASH_PAID = ("cash_paid", "amount_paid_in_cash", "cash_received")
INVOICE_CHANGE = ("change", "change_due", "cash_change")
INVOICE_TAX_INCLUSIVE = (
    "tax_inclusive",
    "tax_is_included",
    "tax_included_in_total",
    "gst_inclusive",
)

BALANCE_TOTAL_CAPITAL_LIABILITIES = (
    "total_capital_and_liabilities",
    "total_capital_liabilities",
    "capital_and_liabilities_total",
    "total_liabilities_and_equity",
)
BALANCE_TOTAL_ASSETS = ("total_assets", "assets_total")

PNL_INTEREST_EARNED = ("interest_earned",)
PNL_OTHER_INCOME = ("other_income",)
PNL_TOTAL_INCOME = ("total_income",)
PNL_INTEREST_EXPENDED = ("interest_expended", "interest_expense")
PNL_OPERATING_EXPENSES = ("operating_expenses",)
PNL_PROVISIONS = ("provisions_and_contingencies", "provisions_contingencies")
PNL_TOTAL_EXPENDITURE = ("total_expenditure", "total_expenses")
PNL_BEFORE_MINORITY = (
    "consolidated_net_profit_before_minority_interest",
    "profit_before_minority_interest",
)
PNL_MINORITY_INTEREST = ("minority_interest",)
PNL_ATTRIBUTABLE = (
    "consolidated_net_profit_attributable_to_the_group",
    "net_profit_attributable_to_the_group",
)
PNL_CURRENT_PROFIT = ("current_profit", "profit_for_current_period")
PNL_BROUGHT_FORWARD = ("brought_forward_profit", "profit_brought_forward")
PNL_AVAILABLE_APPROPRIATION = (
    "total_available_for_appropriation",
    "amount_available_for_appropriation",
)

CASH_OPERATING = (
    "net_cash_flow_from_operating_activities",
    "net_cash_from_operating_activities",
)
CASH_INVESTING = (
    "net_cash_flow_from_investing_activities",
    "net_cash_from_investing_activities",
)
CASH_FINANCING = (
    "net_cash_flow_from_financing_activities",
    "net_cash_from_financing_activities",
)
CASH_FX = (
    "fx_translation_adjustment",
    "foreign_exchange_translation_adjustment",
    "effect_of_exchange_rate_changes",
)
CASH_NET_INCREASE = (
    "net_increase_in_cash_and_cash_equivalents",
    "net_increase_in_cash",
)
CASH_OPENING = (
    "opening_cash_and_cash_equivalents",
    "cash_and_cash_equivalents_at_beginning_of_period",
    "opening_cash",
)
CASH_ADJUSTMENTS = (
    "cash_acquired_on_amalgamation",
    "cash_and_cash_equivalents_on_amalgamation",
    "other_cash_adjustments",
    "other_applicable_adjustments",
)
CASH_CLOSING = (
    "closing_cash_and_cash_equivalents",
    "cash_and_cash_equivalents_at_end_of_period",
    "closing_cash",
)


def parse_financial_value(value: object) -> Decimal:
    """Parse a supported financial value without guessing malformed input."""
    if isinstance(value, bool) or value is None or value == "NOT_APPLICABLE":
        raise ValueError("Value is not a financial number.")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(str(value))
    elif isinstance(value, date):
        raise ValueError("A date is not a financial number.")
    elif isinstance(value, str):
        number = _parse_financial_string(value)
    else:
        raise ValueError("Unsupported financial value type.")

    if not number.is_finite():
        raise ValueError("Financial values must be finite.")
    return number


def validate_financial_data(
    structured_extraction: DocumentStructuredExtraction,
    *,
    tolerance: Decimal | str | int | None = None,
) -> FinancialValidationResult:
    """Run the case-study financial checks for one structured document."""
    resolved_tolerance = _resolve_tolerance(tolerance)
    accessor = _FieldAccessor(structured_extraction)

    validators = {
        "invoice": _validate_invoice,
        "balance_sheet": _validate_balance_sheet,
        "profit_and_loss": _validate_profit_and_loss,
        "cash_flow_statement": _validate_cash_flow,
    }
    checks = validators[structured_extraction.document_type](
        accessor,
        resolved_tolerance,
    )
    status = _aggregate_status(checks)
    return FinancialValidationResult(
        document_type=structured_extraction.document_type,
        tolerance=resolved_tolerance,
        status=status,
        checks=checks,
    )


# A short alias keeps the public service name natural for callers/tests.
validate_financials = validate_financial_data


def _parse_financial_string(raw_value: str) -> Decimal:
    text = raw_value.strip().replace("−", "-").replace("–", "-")
    if not text:
        raise ValueError("Financial value is blank.")

    negative_by_brackets = False
    bracket_pairs = {"(": ")", "[": "]"}
    if text[0] in bracket_pairs:
        if text[-1] != bracket_pairs[text[0]]:
            raise ValueError("Financial value has unmatched brackets.")
        negative_by_brackets = True
        text = text[1:-1].strip()
    elif text[-1:] in bracket_pairs.values():
        raise ValueError("Financial value has unmatched brackets.")

    sign = Decimal("-1") if negative_by_brackets else Decimal("1")
    had_leading_sign = False
    if text[:1] in {"+", "-"}:
        had_leading_sign = True
        if text[0] == "-":
            sign *= -1
        text = text[1:].strip()

    had_currency_symbol = False
    if text[:1] in _CURRENCY_SYMBOLS:
        had_currency_symbol = True
        text = text[1:].strip()

    # Accept the common "$-1,000" layout as well as "-$1,000", but reject
    # duplicate/conflicting signs rather than guessing their intent.
    if text[:1] in {"+", "-"}:
        if negative_by_brackets or had_leading_sign or not had_currency_symbol:
            raise ValueError("Financial value contains multiple signs.")
        if text[0] == "-":
            sign *= -1
        text = text[1:].strip()

    if not _NUMBER_PATTERN.fullmatch(text):
        raise ValueError("Financial value has an unsupported or malformed format.")

    try:
        return sign * Decimal(text.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("Financial value could not be parsed.") from exc


def _resolve_tolerance(value: Decimal | str | int | None) -> Decimal:
    try:
        tolerance = (
            settings.financial_validation_tolerance
            if value is None
            else Decimal(str(value))
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Financial validation tolerance must be a decimal.") from exc
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError(
            "Financial validation tolerance must be finite and non-negative."
        )
    return tolerance


def _resolve_extracted_field(item: ExtractedField) -> _ResolvedValue:
    if item.value == "NOT_APPLICABLE":
        return _ResolvedValue(
            state="not_applicable",
            source_field=item.field,
        )
    try:
        value = parse_financial_value(item.value)
    except ValueError:
        return _ResolvedValue(state="invalid", source_field=item.field)
    return _ResolvedValue(
        state="available",
        value=value,
        source_field=item.field,
    )


def _match_alias_and_period(
    field_name: str,
    alias: str,
) -> tuple[bool, str | None]:
    if field_name == alias:
        return True, None
    prefix = f"{alias}_"
    if field_name.startswith(prefix):
        suffix = field_name[len(prefix) :]
        if _looks_like_period(suffix):
            return True, suffix
    return False, None


def _looks_like_period(value: str) -> bool:
    if value in _PERIOD_WORDS:
        return True
    if re.search(r"(?:^|_)(?:19|20)\d{2}(?:_|$)", value):
        return True
    return bool(
        re.fullmatch(
            r"(?:as_at_)?\d{1,2}_(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"january|february|march|april|june|july|august|september|october|"
            r"november|december)_\d{2}",
            value,
        )
    )


def _periods_for(
    accessor: _FieldAccessor,
    alias_groups: Iterable[Sequence[str]],
) -> list[str | None]:
    periods: set[str | None] = set()
    for aliases in alias_groups:
        periods.update(accessor.periods_for(aliases))
    if not periods:
        return [None]
    return sorted(periods, key=lambda item: (item is not None, item or ""))


def _unavailable(name: str, value: _ResolvedValue) -> str:
    return f"{name} ({value.state})"


def _not_applicable_check(
    *,
    rule_name: str,
    formula: str,
    period: str | None,
    unavailable_inputs: Iterable[str],
) -> FinancialValidationCheck:
    return FinancialValidationCheck(
        rule_name=rule_name,
        period=period,
        formula=formula,
        input_values={},
        calculated_value=None,
        reported_value=None,
        variance=None,
        status="NOT_APPLICABLE",
        unavailable_inputs=list(unavailable_inputs),
    )


def _completed_check(
    *,
    rule_name: str,
    formula: str,
    period: str | None,
    input_values: dict[str, Decimal],
    calculated_value: Decimal,
    reported_value: Decimal,
    tolerance: Decimal,
) -> FinancialValidationCheck:
    variance = abs(calculated_value - reported_value)
    return FinancialValidationCheck(
        rule_name=rule_name,
        period=period,
        formula=formula,
        input_values=input_values,
        calculated_value=calculated_value,
        reported_value=reported_value,
        variance=variance,
        status="PASS" if variance <= tolerance else "FAIL",
        unavailable_inputs=[],
    )


def _sum_rule_checks(
    accessor: _FieldAccessor,
    *,
    rule_name: str,
    formula: str,
    operands: Sequence[_Operand],
    reported_name: str,
    reported_aliases: Sequence[str],
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    alias_groups = [operand.aliases for operand in operands]
    alias_groups.append(reported_aliases)
    checks: list[FinancialValidationCheck] = []

    for period in _periods_for(accessor, alias_groups):
        resolved_operands = [
            (operand, accessor.resolve(operand.aliases, period))
            for operand in operands
        ]
        reported = accessor.resolve(reported_aliases, period)
        unavailable: list[str] = []
        input_values: dict[str, Decimal] = {}
        calculated = Decimal("0")

        for operand, resolved in resolved_operands:
            if (
                resolved.state == "not_applicable"
                and operand.optional_when_explicitly_not_applicable
            ):
                continue
            if resolved.state != "available" or resolved.value is None:
                unavailable.append(_unavailable(operand.name, resolved))
                continue
            input_values[operand.name] = resolved.value
            calculated += operand.coefficient * resolved.value

        if reported.state != "available" or reported.value is None:
            unavailable.append(_unavailable(reported_name, reported))

        if unavailable:
            checks.append(
                _not_applicable_check(
                    rule_name=rule_name,
                    formula=formula,
                    period=period,
                    unavailable_inputs=unavailable,
                )
            )
            continue

        input_values[reported_name] = reported.value
        checks.append(
            _completed_check(
                rule_name=rule_name,
                formula=formula,
                period=period,
                input_values=input_values,
                calculated_value=calculated,
                reported_value=reported.value,
                tolerance=tolerance,
            )
        )
    return checks


def _validate_invoice(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    checks, line_totals = _invoice_line_item_checks(accessor, tolerance)
    checks.append(_invoice_line_totals_check(accessor, line_totals, tolerance))
    checks.append(_invoice_tax_check(accessor, tolerance))
    checks.extend(
        _sum_rule_checks(
            accessor,
            rule_name="invoice_cash_change_reconciliation",
            formula="cash_paid - total_amount = change",
            operands=(
                _Operand("cash_paid", INVOICE_CASH_PAID),
                _Operand("total_amount", INVOICE_TOTAL, Decimal("-1")),
            ),
            reported_name="change",
            reported_aliases=INVOICE_CHANGE,
            tolerance=tolerance,
        )
    )
    return checks


_LINE_ITEM_PATTERN = re.compile(
    r"^(?:line_item|line_items|item)_(?P<item>[a-z0-9]+)_"
    r"(?P<metric>quantity|qty|unit_price|price|line_total|line_amount|total)$"
)
_LINE_ITEM_METRICS = {
    "quantity": "quantity",
    "qty": "quantity",
    "unit_price": "unit_price",
    "price": "unit_price",
    "line_total": "line_total",
    "line_amount": "line_total",
    "total": "line_total",
}


def _invoice_line_item_checks(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> tuple[list[FinancialValidationCheck], list[_ResolvedValue]]:
    items: dict[str, dict[str, ExtractedField]] = {}
    for field in accessor.fields:
        match = _LINE_ITEM_PATTERN.fullmatch(field.field)
        if match:
            metric = _LINE_ITEM_METRICS[match.group("metric")]
            items.setdefault(match.group("item"), {})[metric] = field

    if not items:
        return (
            [
                _not_applicable_check(
                    rule_name="invoice_line_item_calculation",
                    formula="quantity * unit_price = line_total",
                    period=None,
                    unavailable_inputs=["structured_line_items (missing)"],
                )
            ],
            [],
        )

    checks: list[FinancialValidationCheck] = []
    all_line_totals: list[_ResolvedValue] = []
    for item_id in sorted(items):
        metrics = items[item_id]
        resolved = {
            name: (
                _resolve_extracted_field(metrics[name])
                if name in metrics
                else _ResolvedValue(state="missing")
            )
            for name in ("quantity", "unit_price", "line_total")
        }
        all_line_totals.append(resolved["line_total"])
        unavailable = [
            _unavailable(name, value)
            for name, value in resolved.items()
            if value.state != "available"
        ]
        if unavailable:
            checks.append(
                _not_applicable_check(
                    rule_name=f"invoice_line_item_{item_id}",
                    formula="quantity * unit_price = line_total",
                    period=None,
                    unavailable_inputs=unavailable,
                )
            )
            continue

        quantity = resolved["quantity"].value
        unit_price = resolved["unit_price"].value
        line_total = resolved["line_total"].value
        assert quantity is not None and unit_price is not None and line_total is not None
        checks.append(
            _completed_check(
                rule_name=f"invoice_line_item_{item_id}",
                formula="quantity * unit_price = line_total",
                period=None,
                input_values={
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_total": line_total,
                },
                calculated_value=quantity * unit_price,
                reported_value=line_total,
                tolerance=tolerance,
            )
        )
    return checks, all_line_totals


def _invoice_line_totals_check(
    accessor: _FieldAccessor,
    line_totals: list[_ResolvedValue],
    tolerance: Decimal,
) -> FinancialValidationCheck:
    formula = "sum(line_totals) = subtotal"
    if not line_totals:
        return _not_applicable_check(
            rule_name="invoice_line_totals_reconciliation",
            formula=formula,
            period=None,
            unavailable_inputs=["line_totals (missing)"],
        )
    unavailable = [
        _unavailable(f"line_total_{index}", value)
        for index, value in enumerate(line_totals, start=1)
        if value.state != "available"
    ]

    reported = accessor.resolve(INVOICE_SUBTOTAL, None)
    tax_inclusive = _invoice_tax_is_inclusive(accessor)
    reported_name = "subtotal"
    if reported.state == "missing" and tax_inclusive is True:
        reported = accessor.resolve(INVOICE_TOTAL, None)
        reported_name = "tax_inclusive_total"
        formula = "sum(line_totals) = tax_inclusive_total"
    if reported.state != "available" or reported.value is None:
        unavailable.append(_unavailable(reported_name, reported))
    if unavailable:
        return _not_applicable_check(
            rule_name="invoice_line_totals_reconciliation",
            formula=formula,
            period=None,
            unavailable_inputs=unavailable,
        )

    numeric_totals = [item.value for item in line_totals]
    assert all(value is not None for value in numeric_totals)
    calculated = sum((value for value in numeric_totals if value is not None), Decimal("0"))
    return _completed_check(
        rule_name="invoice_line_totals_reconciliation",
        formula=formula,
        period=None,
        input_values={
            **{
                f"line_total_{index}": value
                for index, value in enumerate(numeric_totals, start=1)
                if value is not None
            },
            reported_name: reported.value,
        },
        calculated_value=calculated,
        reported_value=reported.value,
        tolerance=tolerance,
    )


def _invoice_tax_is_inclusive(accessor: _FieldAccessor) -> bool | None:
    for alias in INVOICE_TAX_INCLUSIVE:
        for field in accessor.fields:
            if field.field != alias:
                continue
            if isinstance(field.value, bool):
                return field.value
            if isinstance(field.value, str):
                normalized = field.value.strip().casefold()
                if normalized in {"true", "yes", "included", "tax_inclusive"}:
                    return True
                if normalized in {"false", "no", "excluded", "tax_exclusive"}:
                    return False
    return None


def _invoice_tax_check(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> FinancialValidationCheck:
    if _invoice_tax_is_inclusive(accessor) is True:
        return _not_applicable_check(
            rule_name="invoice_tax_reconciliation",
            formula="taxable_amount + tax = total_amount",
            period=None,
            unavailable_inputs=["separate_tax_addition (not_applicable)"],
        )
    return _sum_rule_checks(
        accessor,
        rule_name="invoice_tax_reconciliation",
        formula="taxable_amount + tax = total_amount",
        operands=(
            _Operand("taxable_amount", INVOICE_TAXABLE_AMOUNT),
            _Operand("tax", INVOICE_TAX),
        ),
        reported_name="total_amount",
        reported_aliases=INVOICE_TOTAL,
        tolerance=tolerance,
    )[0]


def _validate_balance_sheet(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    checks = _sum_rule_checks(
        accessor,
        rule_name="balance_sheet_main_equality",
        formula="total_capital_and_liabilities = total_assets",
        operands=(
            _Operand("total_capital_and_liabilities", BALANCE_TOTAL_CAPITAL_LIABILITIES),
        ),
        reported_name="total_assets",
        reported_aliases=BALANCE_TOTAL_ASSETS,
        tolerance=tolerance,
    )
    checks.extend(
        _component_reconciliation_checks(
            accessor,
            rule_name="balance_sheet_capital_liability_components",
            formula="sum(capital_and_liability_components) = total_capital_and_liabilities",
            dynamic_prefix="capital_and_liability_component_",
            required_components=(
                ("capital", ("capital", "share_capital")),
                ("reserves_and_surplus", ("reserves_and_surplus",)),
                ("deposits", ("deposits",)),
                ("borrowings", ("borrowings",)),
                (
                    "other_liabilities_and_provisions",
                    ("other_liabilities_and_provisions",),
                ),
            ),
            optional_components=(
                ("minority_interest", ("minority_interest",)),
            ),
            reported_aliases=BALANCE_TOTAL_CAPITAL_LIABILITIES,
            reported_name="total_capital_and_liabilities",
            tolerance=tolerance,
        )
    )
    checks.extend(
        _component_reconciliation_checks(
            accessor,
            rule_name="balance_sheet_asset_components",
            formula="sum(asset_components) = total_assets",
            dynamic_prefix="asset_component_",
            required_components=(
                (
                    "cash_and_reserve_bank_balances",
                    (
                        "cash_and_balances_with_reserve_bank_of_india",
                        "cash_and_central_bank_balances",
                    ),
                ),
                (
                    "bank_balances_and_money_at_call",
                    (
                        "balances_with_banks_and_money_at_call_and_short_notice",
                        "bank_balances_and_money_at_call",
                    ),
                ),
                ("investments", ("investments",)),
                ("advances", ("advances",)),
                ("fixed_assets", ("fixed_assets",)),
                ("other_assets", ("other_assets",)),
            ),
            optional_components=(
                ("goodwill_on_consolidation", ("goodwill_on_consolidation",)),
            ),
            reported_aliases=BALANCE_TOTAL_ASSETS,
            reported_name="total_assets",
            tolerance=tolerance,
        )
    )
    return checks


def _component_reconciliation_checks(
    accessor: _FieldAccessor,
    *,
    rule_name: str,
    formula: str,
    dynamic_prefix: str,
    required_components: Sequence[tuple[str, Sequence[str]]],
    optional_components: Sequence[tuple[str, Sequence[str]]],
    reported_aliases: Sequence[str],
    reported_name: str,
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    alias_groups = [aliases for _, aliases in required_components]
    alias_groups.extend(aliases for _, aliases in optional_components)
    alias_groups.append(reported_aliases)
    periods = set(_periods_for(accessor, alias_groups))
    periods.update(_dynamic_component_periods(accessor, dynamic_prefix))
    checks: list[FinancialValidationCheck] = []

    for period in sorted(periods, key=lambda item: (item is not None, item or "")):
        dynamic = _dynamic_components(accessor, dynamic_prefix, period)
        components: dict[str, _ResolvedValue]
        if dynamic:
            components = dynamic
        else:
            components = {
                name: accessor.resolve(aliases, period)
                for name, aliases in required_components
            }
            for name, aliases in optional_components:
                resolved = accessor.resolve(aliases, period)
                if resolved.state in {"available", "invalid"}:
                    components[name] = resolved

        reported = accessor.resolve(reported_aliases, period)
        unavailable = [
            _unavailable(name, value)
            for name, value in components.items()
            if value.state != "available"
        ]
        if not components:
            unavailable.append("components (missing)")
        if reported.state != "available" or reported.value is None:
            unavailable.append(_unavailable(reported_name, reported))
        if unavailable:
            checks.append(
                _not_applicable_check(
                    rule_name=rule_name,
                    formula=formula,
                    period=period,
                    unavailable_inputs=unavailable,
                )
            )
            continue

        values = {
            name: value.value
            for name, value in components.items()
            if value.value is not None
        }
        calculated = sum(values.values(), Decimal("0"))
        values[reported_name] = reported.value
        checks.append(
            _completed_check(
                rule_name=rule_name,
                formula=formula,
                period=period,
                input_values=values,
                calculated_value=calculated,
                reported_value=reported.value,
                tolerance=tolerance,
            )
        )
    return checks


def _dynamic_component_periods(
    accessor: _FieldAccessor,
    prefix: str,
) -> set[str | None]:
    return {
        period
        for field in accessor.fields
        if field.field.startswith(prefix)
        for _, period in [_split_dynamic_component(field.field[len(prefix) :])]
    }


def _dynamic_components(
    accessor: _FieldAccessor,
    prefix: str,
    period: str | None,
) -> dict[str, _ResolvedValue]:
    components: dict[str, _ResolvedValue] = {}
    for field in accessor.fields:
        if not field.field.startswith(prefix):
            continue
        component, component_period = _split_dynamic_component(
            field.field[len(prefix) :]
        )
        if component_period == period and component:
            components[component] = _resolve_extracted_field(field)
    return components


def _split_dynamic_component(value: str) -> tuple[str, str | None]:
    period_match = re.search(
        r"_(?P<period>(?:fy_)?(?:19|20)\d{2}(?:_(?:19|20)?\d{2})?|"
        r"(?:as_at_)?\d{1,2}_[a-z]+_\d{2,4}|current_period|previous_period|"
        r"prior_period|comparative_period|current_year|previous_year|prior_year)$",
        value,
    )
    if period_match and _looks_like_period(period_match.group("period")):
        return value[: period_match.start()], period_match.group("period")
    return value, None


def _validate_profit_and_loss(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    specifications = (
        (
            "profit_and_loss_income_reconciliation",
            "interest_earned + other_income = total_income",
            (
                _Operand("interest_earned", PNL_INTEREST_EARNED),
                _Operand("other_income", PNL_OTHER_INCOME),
            ),
            "total_income",
            PNL_TOTAL_INCOME,
        ),
        (
            "profit_and_loss_expenditure_reconciliation",
            "interest_expended + operating_expenses + provisions_and_contingencies = total_expenditure",
            (
                _Operand("interest_expended", PNL_INTEREST_EXPENDED),
                _Operand("operating_expenses", PNL_OPERATING_EXPENSES),
                _Operand("provisions_and_contingencies", PNL_PROVISIONS),
            ),
            "total_expenditure",
            PNL_TOTAL_EXPENDITURE,
        ),
        (
            "profit_and_loss_net_profit_before_minority",
            "total_income - total_expenditure = consolidated_net_profit_before_minority_interest",
            (
                _Operand("total_income", PNL_TOTAL_INCOME),
                _Operand("total_expenditure", PNL_TOTAL_EXPENDITURE, Decimal("-1")),
            ),
            "profit_before_minority_interest",
            PNL_BEFORE_MINORITY,
        ),
        (
            "profit_and_loss_minority_interest_reconciliation",
            "profit_before_minority_interest - minority_interest = net_profit_attributable_to_the_group",
            (
                _Operand("profit_before_minority_interest", PNL_BEFORE_MINORITY),
                _Operand("minority_interest", PNL_MINORITY_INTEREST, Decimal("-1")),
            ),
            "net_profit_attributable_to_the_group",
            PNL_ATTRIBUTABLE,
        ),
        (
            "profit_and_loss_appropriation_reconciliation",
            "current_profit + brought_forward_profit = total_available_for_appropriation",
            (
                _Operand("current_profit", PNL_CURRENT_PROFIT),
                _Operand("brought_forward_profit", PNL_BROUGHT_FORWARD),
            ),
            "total_available_for_appropriation",
            PNL_AVAILABLE_APPROPRIATION,
        ),
    )
    checks: list[FinancialValidationCheck] = []
    for rule_name, formula, operands, reported_name, aliases in specifications:
        checks.extend(
            _sum_rule_checks(
                accessor,
                rule_name=rule_name,
                formula=formula,
                operands=operands,
                reported_name=reported_name,
                reported_aliases=aliases,
                tolerance=tolerance,
            )
        )
    return checks


def _validate_cash_flow(
    accessor: _FieldAccessor,
    tolerance: Decimal,
) -> list[FinancialValidationCheck]:
    checks = _sum_rule_checks(
        accessor,
        rule_name="cash_flow_net_increase_reconciliation",
        formula=(
            "operating_cash_flow + investing_cash_flow + financing_cash_flow "
            "+ fx_translation_adjustment = net_increase_in_cash"
        ),
        operands=(
            _Operand("operating_cash_flow", CASH_OPERATING),
            _Operand("investing_cash_flow", CASH_INVESTING),
            _Operand("financing_cash_flow", CASH_FINANCING),
            _Operand(
                "fx_translation_adjustment",
                CASH_FX,
                optional_when_explicitly_not_applicable=True,
            ),
        ),
        reported_name="net_increase_in_cash",
        reported_aliases=CASH_NET_INCREASE,
        tolerance=tolerance,
    )
    checks.extend(
        _sum_rule_checks(
            accessor,
            rule_name="cash_flow_opening_closing_reconciliation",
            formula=(
                "opening_cash + net_increase_in_cash + applicable_adjustments "
                "= closing_cash"
            ),
            operands=(
                _Operand("opening_cash", CASH_OPENING),
                _Operand("net_increase_in_cash", CASH_NET_INCREASE),
                _Operand(
                    "applicable_adjustments",
                    CASH_ADJUSTMENTS,
                    optional_when_explicitly_not_applicable=True,
                ),
            ),
            reported_name="closing_cash",
            reported_aliases=CASH_CLOSING,
            tolerance=tolerance,
        )
    )
    return checks


def _aggregate_status(
    checks: Sequence[FinancialValidationCheck],
) -> Literal["PASS", "FAIL", "NOT_APPLICABLE"]:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "PASS" for check in checks):
        return "PASS"
    return "NOT_APPLICABLE"
