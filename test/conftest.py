import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import app
from predict import LoanPredictor

# Load predictor for tests
if app.predictor is None:
    app.predictor = LoanPredictor()


@pytest.fixture
def client():
    return TestClient(app.app)
