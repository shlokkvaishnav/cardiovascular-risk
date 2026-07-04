from src.db.database import _normalize_database_url


def test_normalizes_legacy_postgres_scheme():
    """Heroku-style legacy 'postgres://' scheme (still emitted by some
    managed providers) must be upgraded to the psycopg3 driver scheme."""
    result = _normalize_database_url("postgres://user:pass@host:5432/db")
    assert result == "postgresql+psycopg://user:pass@host:5432/db"


def test_normalizes_bare_postgresql_scheme():
    """A bare 'postgresql://' (SQLAlchemy's default) would silently try to
    use psycopg2, which isn't installed -- must be rewritten to psycopg3."""
    result = _normalize_database_url("postgresql://user:pass@host:5432/db")
    assert result == "postgresql+psycopg://user:pass@host:5432/db"


def test_leaves_explicit_driver_scheme_unchanged():
    url = "postgresql+psycopg://user:pass@host:5432/db"
    assert _normalize_database_url(url) == url


def test_preserves_query_params_and_special_characters():
    url = "postgres://user:p%40ss@host:5432/db?sslmode=require"
    result = _normalize_database_url(url)
    assert result == "postgresql+psycopg://user:p%40ss@host:5432/db?sslmode=require"
