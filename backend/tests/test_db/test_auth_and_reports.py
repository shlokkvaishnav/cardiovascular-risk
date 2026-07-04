"""
Auth + report-history CRUD tests, using an in-memory SQLite database
(overriding the real Postgres dependency) so these run fast and without
needing a live database -- a standard pattern for testing FastAPI apps with
a SQLAlchemy layer. Report-saving mocks the ML prediction call so this
suite doesn't depend on a trained model being loaded.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.database import Base, get_db
from src.api.app import app

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
def _fresh_db():
    """Recreate all tables before each test for isolation."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


VALID_INPUTS = {
    "age": 58, "sex": 1, "height": 175, "weight": 85, "ap_hi": 145, "ap_lo": 90,
    "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
}

FAKE_PREDICTION = (1, 0.83, "High", [{"num__ap_hi": 0.7}], 0.33)


class TestAuth:
    def test_register_creates_user_and_returns_token(self, client):
        res = client.post("/auth/register", json={"email": "a@example.com", "password": "password123"})
        assert res.status_code == 201
        assert "access_token" in res.json()

    def test_register_duplicate_email_returns_409(self, client):
        client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
        res = client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
        assert res.status_code == 409

    def test_login_with_correct_credentials_succeeds(self, client):
        client.post("/auth/register", json={"email": "b@example.com", "password": "password123"})
        res = client.post("/auth/login", json={"email": "b@example.com", "password": "password123"})
        assert res.status_code == 200
        assert "access_token" in res.json()

    def test_login_with_wrong_password_fails(self, client):
        client.post("/auth/register", json={"email": "c@example.com", "password": "password123"})
        res = client.post("/auth/login", json={"email": "c@example.com", "password": "wrongpassword"})
        assert res.status_code == 401

    def test_me_requires_valid_token(self, client):
        assert client.get("/auth/me").status_code == 401
        assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401

    def test_me_returns_current_user_with_valid_token(self, client):
        token = client.post("/auth/register", json={"email": "d@example.com", "password": "password123"}).json()["access_token"]
        res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json()["email"] == "d@example.com"


class TestReports:
    def _auth_headers(self, client, email="reports@example.com"):
        token = client.post("/auth/register", json={"email": email, "password": "password123"}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_save_report_requires_auth(self, client):
        res = client.post("/reports", json={"inputs": VALID_INPUTS})
        assert res.status_code == 401

    @patch("src.api.reports._run_prediction", return_value=FAKE_PREDICTION)
    def test_save_and_list_reports(self, mock_predict, client):
        headers = self._auth_headers(client)
        save_res = client.post("/reports", json={"inputs": VALID_INPUTS, "note": "test"}, headers=headers)
        assert save_res.status_code == 201
        body = save_res.json()
        assert body["risk_level"] == "High"
        assert body["prediction"] == 1

        list_res = client.get("/reports", headers=headers)
        assert list_res.status_code == 200
        assert len(list_res.json()) == 1

    @patch("src.api.reports._run_prediction", return_value=FAKE_PREDICTION)
    def test_get_report_by_id(self, mock_predict, client):
        headers = self._auth_headers(client)
        report_id = client.post("/reports", json={"inputs": VALID_INPUTS}, headers=headers).json()["id"]

        res = client.get(f"/reports/{report_id}", headers=headers)
        assert res.status_code == 200
        assert res.json()["inputs"]["age"] == 58

    @patch("src.api.reports._run_prediction", return_value=FAKE_PREDICTION)
    def test_cannot_access_another_users_report(self, mock_predict, client):
        headers_a = self._auth_headers(client, "usera@example.com")
        headers_b = self._auth_headers(client, "userb@example.com")

        report_id = client.post("/reports", json={"inputs": VALID_INPUTS}, headers=headers_a).json()["id"]

        res = client.get(f"/reports/{report_id}", headers=headers_b)
        assert res.status_code == 404

    def test_get_nonexistent_report_returns_404(self, client):
        headers = self._auth_headers(client)
        res = client.get("/reports/does-not-exist", headers=headers)
        assert res.status_code == 404
