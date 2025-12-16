"""
API Integration Tests
=====================

Integration tests for prediction API endpoints.

Run with: pytest tests/test_api.py -v
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Check if FastAPI and dependencies are available
try:
    from fastapi.testclient import TestClient
    from api.prediction_api import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPredictionAPI:
    """Integration tests for prediction API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_weekly_predictions_no_model(self, client):
        """Test weekly predictions when model not trained."""
        response = client.get("/api/predictions/weekly")
        
        # Should return 503 if models not trained
        # or 200 if models are available
        assert response.status_code in [200, 503, 404]
    
    def test_spike_probability_no_model(self, client):
        """Test spike probability when model not trained."""
        response = client.get("/api/predictions/spike-probability")
        
        assert response.status_code in [200, 503, 404]
    
    def test_trend_analysis(self, client):
        """Test trend analysis endpoint."""
        response = client.get("/api/analytics/trends?period=30")
        
        # May return 404 if no data, or 200 with data
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "period_days" in data
            assert "sentiment_trend" in data
            assert "volume_trend" in data
    
    def test_trend_analysis_invalid_period(self, client):
        """Test trend analysis with invalid period."""
        response = client.get("/api/analytics/trends?period=1000")
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_daily_aggregates(self, client):
        """Test daily aggregates endpoint."""
        response = client.get("/api/data/daily-aggregates?days=7")
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_prediction_history(self, client):
        """Test prediction history endpoint."""
        response = client.get("/api/predictions/history?days=7")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 7
    
    def test_retrain_without_api_key(self, client):
        """Test retrain endpoint without API key in production."""
        # In development mode, this may succeed
        # In production, it should require API key
        response = client.post("/api/models/retrain")
        
        # Accept either success (dev mode) or auth required (prod mode) or data not found
        assert response.status_code in [200, 403, 404]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIResponseSchemas:
    """Test API response schemas match expected format."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_response_schema(self, client):
        """Test health endpoint response schema."""
        response = client.get("/api/health")
        data = response.json()
        
        required_fields = ["status", "timestamp", "models_loaded", "data_loaded"]
        for field in required_fields:
            assert field in data
    
    def test_prediction_history_schema(self, client):
        """Test prediction history response schema."""
        response = client.get("/api/predictions/history?days=3")
        data = response.json()
        
        assert len(data) == 3
        
        for item in data:
            assert "date" in item
            assert "predicted_sentiment" in item
            assert "actual_sentiment" in item


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
