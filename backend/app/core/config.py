import os
from dataclasses import dataclass


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


settings = Settings()
