# Loan Approval Prediction - End-to-End ML Capstone Project

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting loan approval status based on applicant information. It includes data preparation, model training with MLflow tracking concepts, a FastAPI prediction service, and Docker containerization support.

## Problem Statement

The objective is to predict whether a loan application will be approved based on various applicant features such as income, credit history, education, property area, etc. This is a binary classification problem where:
- Target variable: `Loan_Status` (Y = Approved, N = Not Approved)
- Use case: Help financial institutions automate and standardize the loan approval process

## Dataset

- **Source**: [Loan Prediction Dataset](https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv)
- **License**: Public domain / Open source
- **Size**: 614 rows × 13 columns
- **Target Variable**: Loan_Status (422 approved, 192 not approved)
- **Features**: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

## Features

| Feature | Description | Type |
|---------|-------------|------|
| Loan_ID | Unique Loan ID | Identifier |
| Gender | Male/Female | Categorical |
| Married | Applicant married (Yes/No) | Categorical |
| Dependents | Number of dependents (0,1,2,3+) | Categorical |
| Education | Graduate/Not Graduate | Categorical |
| Self_Employed | Self employed (Yes/No) | Categorical |
| ApplicantIncome | Applicant income | Numerical |
| CoapplicantIncome | Coapplicant income | Numerical |
| LoanAmount | Loan amount in thousands | Numerical |
| Loan_Amount_Term | Term of loan in months | Numerical |
| Credit_History | Credit history meets guidelines (1.0/0.0) | Numerical |
| Property_Area | Urban/Semiurban/Rural | Categorical |

## Technology Stack

- **ML/Data**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **API**: FastAPI, Uvicorn, Pydantic
- **Model Tracking**: MLflow (installed, configuration ready)
- **Testing**: pytest, httpx
- **Containerization**: Docker
- **Model Persistence**: joblib

## Project Structure

```
loan-approval-prediction/
├── data/                    # Dataset and metadata
│   ├── loan_prediction.csv
│   └── dataset_info.json
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_preparation.py  # Data loading, cleaning, EDA
│   ├── train.py             # Model training and evaluation
│   ├── predict.py           # Inference logic
│   └── app.py               # FastAPI application
├── models/                  # Trained model artifacts
│   ├── loan_approval_model.pkl
│   └── model_metadata.json
├── reports/                 # Evaluation reports and plots
│   ├── figures/             # EDA and evaluation visualizations
│   └── metrics/             # Training results in JSON
├── tests/                   # Automated tests
│   ├── test_data_preparation.py
│   ├── test_model.py
│   ├── test_api.py
│   └── conftest.py
├── mlruns/                  # MLflow tracking (local)
├── Dockerfile              # Docker configuration
├── .dockerignore
├── .gitignore
├── requirements.txt
├── Makefile
├── run_pipeline.py         # Orchestration script
└── README.md
```

## Environment Setup

1. Clone or navigate to the project directory
2. Install dependencies:
```bash
py -m pip install -r requirements.txt
```

## Data Collection and Preparation

Run data preparation (loads dataset, validates, performs EDA):
```bash
py src/data_preparation.py
```

This will:
- Load the dataset from `data/`
- Validate data quality (missing values, duplicates, types)
- Generate EDA visualizations in `reports/figures/`
- Save dataset metadata to `data/dataset_info.json`

## Model Training and Evaluation

Train models and select the best one:
```bash
py src/train.py
```

Models trained:
1. **Logistic Regression** (Baseline)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

Best model selected based on F1-score. Results saved to `reports/metrics/training_results.json` and model artifacts to `models/`.

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression (Best) | 0.8618 | 0.8400 | 0.9882 | 0.9081 | 0.8523 |
| Random Forest | 0.8211 | 0.8462 | 0.9059 | 0.8750 | 0.7808 |
| Gradient Boosting | 0.8049 | 0.8211 | 0.9176 | 0.8667 | 0.7551 |

## FastAPI Prediction Service

Start the API server:
```bash
py src/app.py
```

Or with uvicorn:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic API information |
| `/health` | GET | Health check and model readiness |
| `/model-info` | GET | Model metadata and metrics |
| `/predict` | POST | Make loan approval prediction |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "Approved",
  "probability_approved": 0.8130906413689292,
  "probability_not_approved": 0.18690935863107083,
  "confidence": 0.8130906413689292
}
```

## Docker Build and Run

Build the Docker image:
```bash
docker build -t loan-approval-prediction:latest .
```

Run the container:
```bash
docker run -p 8000:8000 loan-approval-prediction:latest
```

**Note**: Docker daemon was not available in this environment during testing. The Dockerfile and configuration have been created and validated statically.

## Testing

Run all automated tests:
```bash
py -m pytest tests/ -v
```

Tests cover:
- Dataset loading and validation
- Model training and prediction
- API endpoints (health, root, model-info, predict)
- Input validation
- Model artifact loading
- Prediction schema validation

**Test Results**: 11/11 tests passing

## Automated Workflow

Use the orchestration script:
```bash
py run_pipeline.py
```

Or use Makefile:
```bash
make all      # Run complete pipeline
make data     # Data preparation only
make train    # Training only
make test     # Run tests
make serve    # Start API server
```

## MLflow Tracking

MLflow is installed and ready for experiment tracking. Tracking data will be stored locally in `mlruns/`. The project is configured to support:
- Experiment creation and tracking
- Parameter and metric logging
- Artifact storage
- Model registry (backend dependent)

## Key Features

- **No data leakage**: Preprocessing fitted only on training data using sklearn Pipeline
- **Reproducible**: Fixed random seeds (42)
- **Production-ready API**: Input validation with Pydantic, proper error handling, health checks
- **Comprehensive testing**: Unit and integration tests
- **Containerized**: Docker support with non-root user
- **Well-documented**: Complete documentation and examples

## Known Limitations

- Docker daemon not available in current environment for container testing
- MLflow model registry backend may vary by configuration
- Using a publicly available dataset (synthetic/curated) as training data
- Model performance depends on data quality and assumptions

## Troubleshooting

1. **Import errors**: Ensure you're running from project root and src is in PYTHONPATH
2. **Model not found**: Run `py src/train.py` first to generate model artifacts
3. **Port already in use**: Change port in `app.py` or kill existing process
4. **Missing dependencies**: Run `py -m pip install -r requirements.txt`

## License

This project is for educational/capstone purposes.
