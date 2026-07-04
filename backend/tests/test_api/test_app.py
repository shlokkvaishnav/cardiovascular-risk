"""
Comprehensive API tests
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.app import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
        assert "features" in data
        # 6 raw numeric + 5 derived numeric (bmi/pulse_pressure/map_pressure/
        # health_risk_score/bmi_bp_interaction) + 5 raw categorical + 3 derived
        # categorical (bp/bmi category, age bucket)
        assert len(data["features"]) == 19


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    @pytest.fixture
    def valid_request_data(self):
        """Valid prediction request data (Kaggle cardiovascular lifestyle schema)"""
        return {
            "age": 58,
            "sex": 1,
            "height": 175,
            "weight": 85,
            "ap_hi": 145,
            "ap_lo": 90,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        }

    def test_predict_valid_request(self, valid_request_data):
        """Test prediction with valid data"""
        response = client.post(
            "/predict", json=valid_request_data, headers={"X-API-Key": "dev-api-key"}
        )

        # May return 503 if model not loaded, which is acceptable
        if response.status_code == 503:
            assert "Model is not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_returns_signed_shap_contributors(self, valid_request_data):
        """Regression test: /predict must return non-null, signed SHAP
        contributions regardless of which model type won training (the old
        coefficient-proxy explanation silently returned None for
        RandomForest/SVM -- this is the literal bug that was fixed)."""
        response = client.post(
            "/predict", json=valid_request_data, headers={"X-API-Key": "dev-api-key"}
        )

        if response.status_code == 503:
            pytest.skip("Model not loaded in this environment")

        data = response.json()
        assert data["top_contributors"] is not None
        assert len(data["top_contributors"]) > 0
        assert data["baseline_probability"] is not None
        # At least one contribution should carry a sign (not all-zero/abs-only)
        values = [v for entry in data["top_contributors"] for v in entry.values()]
        assert any(v != 0 for v in values)

    def test_predict_invalid_age(self, valid_request_data):
        """Test prediction with invalid age"""
        invalid_data = valid_request_data.copy()
        invalid_data["age"] = 150  # Invalid age

        response = client.post(
            "/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code == 422  # Validation error

    def test_predict_missing_field(self, valid_request_data):
        """Test prediction with missing required field"""
        incomplete_data = valid_request_data.copy()
        del incomplete_data["age"]

        response = client.post(
            "/predict", json=incomplete_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code == 422

    def test_predict_invalid_type(self, valid_request_data):
        """Test prediction with invalid data type"""
        invalid_data = valid_request_data.copy()
        invalid_data["age"] = "not a number"

        response = client.post(
            "/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code == 422

    def test_predict_out_of_range_values(self, valid_request_data):
        """Test prediction with out of range values"""
        invalid_data = valid_request_data.copy()
        invalid_data["weight"] = 1000  # Too high

        response = client.post(
            "/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code == 422

    def test_predict_inconsistent_blood_pressure(self, valid_request_data):
        """Systolic must meaningfully exceed diastolic"""
        invalid_data = valid_request_data.copy()
        invalid_data["ap_hi"] = 80
        invalid_data["ap_lo"] = 90

        response = client.post(
            "/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code == 422


class TestBatchPrediction:
    """Test batch prediction endpoint"""

    @pytest.fixture
    def valid_batch_request(self):
        """Valid batch prediction request"""
        return {
            "instances": [
                {
                    "age": 58,
                    "sex": 1,
                    "height": 175,
                    "weight": 85,
                    "ap_hi": 145,
                    "ap_lo": 90,
                    "cholesterol": 2,
                    "gluc": 1,
                    "smoke": 0,
                    "alco": 0,
                    "active": 1,
                },
                {
                    "age": 45,
                    "sex": 0,
                    "height": 165,
                    "weight": 60,
                    "ap_hi": 110,
                    "ap_lo": 70,
                    "cholesterol": 1,
                    "gluc": 1,
                    "smoke": 0,
                    "alco": 0,
                    "active": 1,
                },
            ]
        }

    def test_batch_predict_valid(self, valid_batch_request):
        """Test batch prediction with valid data"""
        response = client.post(
            "/batch-predict",
            json=valid_batch_request,
            headers={"X-API-Key": "dev-api-key"},
        )

        # May return 503 if model not loaded
        if response.status_code == 503:
            assert "Model is not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert "timestamp" in data

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty instances list"""
        response = client.post(
            "/batch-predict",
            json={"instances": []},
            headers={"X-API-Key": "dev-api-key"},
        )
        assert response.status_code == 422

    def test_batch_predict_too_many_instances(self):
        """Test batch prediction with too many instances"""
        instances = [
            {
                "age": 58,
                "sex": 1,
                "height": 175,
                "weight": 85,
                "ap_hi": 145,
                "ap_lo": 90,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1,
            }
        ] * 101  # More than max allowed (100)

        response = client.post(
            "/batch-predict",
            json={"instances": instances},
            headers={"X-API-Key": "dev-api-key"},
        )
        assert response.status_code == 422


class TestModelManagement:
    """Test model management endpoints"""

    def test_model_reload(self):
        """Test model reload endpoint"""
        response = client.post("/model/reload")

        # May return 404 if model file doesn't exist, which is acceptable
        if response.status_code == 404:
            assert "Model file not found" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "metadata" in data


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method(self):
        """Test using wrong HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


class TestRequestValidation:
    """Test comprehensive request validation"""

    def test_boundary_values(self):
        """Test boundary values for all fields"""
        # Minimum valid values
        min_data = {
            "age": 18,
            "sex": 0,
            "height": 120,
            "weight": 30,
            "ap_hi": 90,
            "ap_lo": 70,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 0,
        }

        response = client.post(
            "/predict", json=min_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code in [200, 503]  # 503 if model not loaded

        # Maximum valid values
        max_data = {
            "age": 100,
            "sex": 1,
            "height": 220,
            "weight": 250,
            "ap_hi": 240,
            "ap_lo": 160,
            "cholesterol": 3,
            "gluc": 3,
            "smoke": 1,
            "alco": 1,
            "active": 1,
        }

        response = client.post(
            "/predict", json=max_data, headers={"X-API-Key": "dev-api-key"}
        )
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
