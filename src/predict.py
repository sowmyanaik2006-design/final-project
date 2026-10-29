import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"

class LoanPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "loan_approval_model.pkl"
        self.model_path = model_path
        self.model = self.load_model()
        self.metadata = self.load_metadata()
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        return joblib.load(self.model_path)
    
    def load_metadata(self):
        """Load model metadata."""
        metadata_path = MODELS_DIR / "model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
    
    def predict(self, input_data):
        """Make prediction on input data."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict or DataFrame")
        
        # Make prediction
        prediction = self.model.predict(df)
        probabilities = self.model.predict_proba(df) if hasattr(self.model, "predict_proba") else None
        
        result = {
            "prediction": int(prediction[0]),
            "prediction_label": "Approved" if prediction[0] == 1 else "Not Approved"
        }
        
        if probabilities is not None:
            result["probability_approved"] = float(probabilities[0][1])
            result["probability_not_approved"] = float(probabilities[0][0])
            result["confidence"] = float(max(probabilities[0]))
        
        return result
    
    def predict_batch(self, input_data):
        """Make batch predictions."""
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be DataFrame")
        
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df) if hasattr(self.model, "predict_proba") else None
        
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": int(predictions[i]),
                "prediction_label": "Approved" if predictions[i] == 1 else "Not Approved"
            }
            if probabilities is not None:
                result["probability_approved"] = float(probabilities[i][1])
                result["probability_not_approved"] = float(probabilities[i][0])
                result["confidence"] = float(max(probabilities[i]))
            results.append(result)
        
        return results


if __name__ == "__main__":
    predictor = LoanPredictor()
    # Test prediction
    test_input = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 150.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    result = predictor.predict(test_input)
    print(result)
