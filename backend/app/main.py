from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging


configure_logging()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
templates = Jinja2Templates(directory=str(FRONTEND_ROOT / "templates"))


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

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_ROOT / "static")),
    name="static",
)
app.include_router(documents_router, prefix=settings.api_v1_prefix)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"request": request},
    )


@app.get(
    "/documents/{document_name}",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def document_result(request: Request, document_name: str):
    return templates.TemplateResponse(
        request=request,
        name="document_result.html",
        context={
            "request": request,
            "document_name": document_name,
        },
    )
