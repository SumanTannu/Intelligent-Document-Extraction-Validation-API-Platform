from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
)

app.include_router(documents_router, prefix=settings.api_v1_prefix)
