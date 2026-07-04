"""
Database connection setup for the optional accounts/report-history feature.

Guest/anonymous usage (the default, zero-storage flow) never touches this
module at all -- it's only imported by the auth/reports routers, which are
opt-in from the frontend's perspective.
"""
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

def _normalize_database_url(url: str) -> str:
    """Managed Postgres providers (Railway, Render, Heroku, ...) typically
    inject DATABASE_URL as a bare 'postgresql://' or legacy 'postgres://'
    string. SQLAlchemy would then default to the psycopg2 driver, which
    isn't installed here (we use psycopg 3 / 'psycopg[binary]') -- rewrite
    the scheme so the psycopg3 driver is used consistently everywhere."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _normalize_database_url(
    os.getenv("DATABASE_URL", "postgresql+psycopg://cardio:cardio@localhost:5432/cardio")
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
