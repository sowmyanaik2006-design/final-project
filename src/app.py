import os
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn

from predict import LoanPredictor

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status",
    version="1.0.0"
)

# Load model at startup
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    try:
        predictor = LoanPredictor()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


class LoanApplication(BaseModel):
    Gender: str = Field(..., description="Gender of applicant (Male/Female)")
    Married: str = Field(..., description="Marital status (Yes/No)")
    Dependents: str = Field(..., description="Number of dependents (0,1,2,3+)")
    Education: str = Field(..., description="Education status (Graduate/Not Graduate)")
    Self_Employed: str = Field(..., description="Self employment status (Yes/No)")
    ApplicantIncome: float = Field(..., gt=0, description="Applicant income")
    CoapplicantIncome: float = Field(default=0.0, ge=0, description="Coapplicant income")
    LoanAmount: float = Field(..., gt=0, description="Loan amount in thousands")
    Loan_Amount_Term: float = Field(..., gt=0, description="Loan amount term in months")
    Credit_History: float = Field(..., description="Credit history (1.0 for good, 0.0 for bad)")
    Property_Area: str = Field(..., description="Property area (Urban/Semiurban/Rural)")
    
    @validator('Gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('Married')
    def validate_married(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Married must be Yes or No')
        return v
    
    @validator('Dependents')
    def validate_dependents(cls, v):
        if v not in ['0', '1', '2', '3+']:
            raise ValueError('Dependents must be 0, 1, 2, or 3+')
        return v
    
    @validator('Education')
    def validate_education(cls, v):
        if v not in ['Graduate', 'Not Graduate']:
            raise ValueError('Education must be Graduate or Not Graduate')
        return v
    
    @validator('Self_Employed')
    def validate_self_employed(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Self_Employed must be Yes or No')
        return v
    
    @validator('Property_Area')
    def validate_property_area(cls, v):
        if v not in ['Urban', 'Semiurban', 'Rural']:
            raise ValueError('Property_Area must be Urban, Semiurban, or Rural')
        return v
    
    @validator('Credit_History')
    def validate_credit_history(cls, v):
        if v not in [0.0, 1.0]:
            raise ValueError('Credit_History must be 0.0 or 1.0')
        return v


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_approved: Optional[float] = None
    probability_not_approved: Optional[float] = None
    confidence: Optional[float] = None


@app.get("/")
async def root():
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/model-info", "/predict"]
    }


@app.get("/health")
async def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": "Loan Approval Prediction Model",
        "model_type": predictor.metadata.get("model_type", "Unknown"),
        "metrics": predictor.metadata.get("metrics", {}),
        "saved_at": predictor.metadata.get("saved_at", "Unknown")
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(loan_app: LoanApplication):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict
        input_data = loan_app.dict()
        result = predictor.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
