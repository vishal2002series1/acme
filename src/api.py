from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Get the directory where the current file (api.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(current_dir), 'models', 'churn_model.joblib')

# Load the model
# MODEL_PATH = './models/churn_model.joblib'
model = joblib.load(MODEL_PATH)

# Define the API
app = FastAPI()

# Define the data model for input validation
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Define the prediction endpoint
@app.post("/predict")
async def predict_churn(customer: Customer):
    """
    Predicts customer churn based on input data.
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([customer.dict()])

        # Preprocess the input data
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']
        for col in categorical_cols:
            input_data[col] = LabelEncoder().fit_transform(input_data[col])

        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        scaler = StandardScaler()
        input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return {"churn_prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))