from fastapi.testclient import TestClient
from api.app import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction_valid_input():
    # Valid normal case
    payload = {
        "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
    }
    # Mocking model presence by expecting either success or 503 (Services Unavailable) 
    # if the physical file isn't there in test env
    response = client.post("/predict", json=payload, headers={"X-API-Key": "cardiovascular-risk-secret-key-123"})
    
    if response.status_code == 503:
        assert response.json()["detail"] == "Model is not loaded"
    else:
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "risk_level" in data

def test_edge_case_blood_pressure():
    # Test BP upper limit (300 is allowed, 301 is not)
    payload = {
        "age": 50, "sex": 1, "cp": 0, "trestbps": 301, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
    }
    response = client.post("/predict", json=payload, headers={"X-API-Key": "cardiovascular-risk-secret-key-123"})
    assert response.status_code == 422
    
def test_edge_case_heart_rate_age():
    # Test Heart Rate vs Age logic (if implemented in schemas)
    # Our schema validator prints warning or error?
    # Current implementation in schemas.py has a validator:
    # if v > max_hr * 1.1: raise ValueError
    
    # Age 60 -> Max HR ~160. 1.1x = 176.
    # Input 220 should fail.
    payload = {
        "age": 60, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 220, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
    }
    response = client.post("/predict", json=payload, headers={"X-API-Key": "cardiovascular-risk-secret-key-123"})
    assert response.status_code == 422

def test_missing_auth():
    response = client.post("/predict", json={})
    assert response.status_code == 403
