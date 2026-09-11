from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by the application's ORM models."""


def _create_engine(database_url: str) -> Engine:
    connect_args = (
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {}
    )
    return create_engine(database_url, connect_args=connect_args)


engine = _create_engine(settings.database_url)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def init_db(bind: Engine | None = None) -> None:
    """Create application tables for the configured database."""
    # Importing registers the model with Base.metadata.
    from app.models.document import Document  # noqa: F401

    Base.metadata.create_all(bind=bind or engine)


def get_db() -> Generator[Session, None, None]:
    """Yield one SQLAlchemy session per FastAPI request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
