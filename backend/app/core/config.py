import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


def _non_negative_decimal_setting(name: str, default: str) -> Decimal:
    raw_value = os.getenv(name, default)
    try:
        value = Decimal(raw_value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal number.") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative decimal number.")
    return value


def _positive_integer_setting(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _comma_separated_setting(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        value.strip().rstrip("/")
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv(
        "APP_TITLE",
        "Intelligent Document Extraction, Validation & API Platform",
    )
    app_description: str = os.getenv(
        "APP_DESCRIPTION",
        "API foundation for uploading and retrieving financial documents.",
    )
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    api_v1_prefix: str = "/api/v1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./intellidoc.sqlite3",
    )
    cors_allowed_origins: tuple[str, ...] = _comma_separated_setting(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    # Keep enough of the developer-tier token budget available for the prompt.
    llm_max_completion_tokens: int = _positive_integer_setting(
        "LLM_MAX_COMPLETION_TOKENS",
        "4096",
    )
    # The case study does not prescribe a numeric tolerance. One currency cent
    # is the explicit default and can be overridden without code changes.
    financial_validation_tolerance: Decimal = _non_negative_decimal_setting(
        "FINANCIAL_VALIDATION_TOLERANCE",
        "0.01",
    )


settings = Settings()
