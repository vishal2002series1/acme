from fastapi.testclient import TestClient
from src.api import app  # Import your FastAPI app
import pytest

client = TestClient(app)

def test_predict_endpoint():
    """
    Tests the /predict endpoint with sample data.
    """
    sample_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 29,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.5,
        "TotalCharges": 2861.4
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert response.json()["churn_prediction"] in [0, 1]

def test_predict_endpoint_invalid_data():
    """
    Tests the /predict endpoint with invalid data.
    """
    invalid_data = {
        "gender": "Invalid",  # Invalid value
        "SeniorCitizen": "abc",  # Invalid type
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 29,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.5,
        "TotalCharges": 2861.4
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Expecting a validation error