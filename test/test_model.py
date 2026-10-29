import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import LoanPredictor


def test_model_loads():
    predictor = LoanPredictor()
    assert predictor.model is not None


def test_prediction():
    predictor = LoanPredictor()
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
    assert "prediction" in result
    assert "prediction_label" in result
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["Approved", "Not Approved"]


def test_prediction_schema():
    predictor = LoanPredictor()
    test_input = {
        "Gender": "Female",
        "Married": "No",
        "Dependents": "1",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 3000,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 100.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 0.0,
        "Property_Area": "Rural"
    }
    result = predictor.predict(test_input)
    assert isinstance(result["prediction"], int)
    assert isinstance(result["prediction_label"], str)
    if "confidence" in result:
        assert 0 <= result["confidence"] <= 1
