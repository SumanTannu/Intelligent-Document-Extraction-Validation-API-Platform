from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging


configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.include_router(documents_router, prefix=settings.api_v1_prefix)


@app.get("/", include_in_schema=False)
def service_info() -> dict[str, str]:
    return {
        "service": settings.app_title,
        "status": "healthy",
        "docs": "/docs",
    }
