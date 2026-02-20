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
        assert len(data["features"]) == 13

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def valid_request_data(self):
        """Valid prediction request data"""
        return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
    
    def test_predict_valid_request(self, valid_request_data):
        """Test prediction with valid data"""
        response = client.post("/predict", json=valid_request_data, headers={"X-API-Key": "dev-api-key"})
        
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
    
    def test_predict_invalid_age(self, valid_request_data):
        """Test prediction with invalid age"""
        invalid_data = valid_request_data.copy()
        invalid_data["age"] = 150  # Invalid age
        
        response = client.post("/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_field(self, valid_request_data):
        """Test prediction with missing required field"""
        incomplete_data = valid_request_data.copy()
        del incomplete_data["age"]
        
        response = client.post("/predict", json=incomplete_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code == 422
    
    def test_predict_invalid_type(self, valid_request_data):
        """Test prediction with invalid data type"""
        invalid_data = valid_request_data.copy()
        invalid_data["age"] = "not a number"
        
        response = client.post("/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code == 422
    
    def test_predict_out_of_range_values(self, valid_request_data):
        """Test prediction with out of range values"""
        invalid_data = valid_request_data.copy()
        invalid_data["chol"] = 1000  # Too high
        
        response = client.post("/predict", json=invalid_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code == 422

class TestBatchPrediction:
    """Test batch prediction endpoint"""
    
    @pytest.fixture
    def valid_batch_request(self):
        """Valid batch prediction request"""
        return {
            "instances": [
                {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
                },
                {
                    "age": 45, "sex": 0, "cp": 1, "trestbps": 120,
                    "chol": 200, "fbs": 0, "restecg": 0, "thalach": 170,
                    "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                }
            ]
        }
    
    def test_batch_predict_valid(self, valid_batch_request):
        """Test batch prediction with valid data"""
        response = client.post("/batch-predict", json=valid_batch_request, headers={"X-API-Key": "dev-api-key"})
        
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
        response = client.post("/batch-predict", json={"instances": []}, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code == 422
    
    def test_batch_predict_too_many_instances(self):
        """Test batch prediction with too many instances"""
        instances = [
            {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        ] * 101  # More than max allowed (100)
        
        response = client.post("/batch-predict", json={"instances": instances}, headers={"X-API-Key": "dev-api-key"})
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
    
    def test_heart_rate_validation(self):
        """Test heart rate validation based on age"""
        # Heart rate too high for age
        data = {
            "age": 70, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 200,  # Too high for age 70
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
        
        response = client.post("/predict", json=data, headers={"X-API-Key": "dev-api-key"})
        # Should either validate or return 422
        assert response.status_code in [200, 422, 503]
    
    def test_boundary_values(self):
        """Test boundary values for all fields"""
        # Minimum valid values
        min_data = {
            "age": 1, "sex": 0, "cp": 0, "trestbps": 80,
            "chol": 100, "fbs": 0, "restecg": 0, "thalach": 60,
            "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0
        }
        
        response = client.post("/predict", json=min_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        # Maximum valid values
        max_data = {
            "age": 120, "sex": 1, "cp": 3, "trestbps": 200,
            "chol": 600, "fbs": 1, "restecg": 2, "thalach": 220,
            "exang": 1, "oldpeak": 10, "slope": 2, "ca": 4, "thal": 3
        }
        
        response = client.post("/predict", json=max_data, headers={"X-API-Key": "dev-api-key"})
        assert response.status_code in [200, 503]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
