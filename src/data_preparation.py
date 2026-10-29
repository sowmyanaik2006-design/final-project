import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = Path(__file__).parent.parent / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_or_load_dataset():
    """Load the loan prediction dataset from local file."""
    data_path = DATA_DIR / "loan_prediction.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure data is present.")
    
    df = pd.read_csv(data_path)
    return df, data_path


def validate_dataset(df):
    """Validate the dataset."""
    validation_report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df["Loan_Status"].value_counts().to_dict() if "Loan_Status" in df.columns else None,
    }
    return validation_report


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data with no leakage - fit on training data only."""
    # Drop Loan_ID as it's not a feature
    df = df.copy()
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)
    
    # Separate features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"].map({"Y": 1, "N": 0})  # Encode target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features


def perform_eda(df):
    """Perform exploratory data analysis and save plots."""
    df_eda = df.copy()
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    df_eda["Loan_Status"].value_counts().plot(kind="bar")
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Missing values
    plt.figure(figsize=(10, 6))
    missing = df_eda.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        missing.plot(kind="bar")
        plt.title("Missing Values by Feature")
        plt.xlabel("Features")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Correlation heatmap for numerical features
    plt.figure(figsize=(10, 8))
    numeric_df = df_eda.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Loan status by education
    if "Education" in df_eda.columns:
        plt.figure(figsize=(8, 6))
        pd.crosstab(df_eda["Education"], df_eda["Loan_Status"]).plot(kind="bar", stacked=True)
        plt.title("Loan Status by Education")
        plt.xlabel("Education")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "loan_by_education.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # Loan status by property area
    if "Property_Area" in df_eda.columns:
        plt.figure(figsize=(8, 6))
        pd.crosstab(df_eda["Property_Area"], df_eda["Loan_Status"]).plot(kind="bar", stacked=True)
        plt.title("Loan Status by Property Area")
        plt.xlabel("Property Area")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "loan_by_property.png", dpi=150, bbox_inches="tight")
        plt.close()


def save_dataset_info(df, validation_report):
    """Save dataset information."""
    info = {
        "dataset_name": "loan_prediction",
        "source": "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv",
        "license": "Public domain / open source",
        "download_date": datetime.now().isoformat(),
        "target_variable": "Loan_Status (Y=Approved, N=Not Approved)",
        "shape": validation_report["shape"],
        "features": [col for col in df.columns if col != "Loan_Status"],
        "validation_report": validation_report,
        "problem_type": "binary_classification",
        "description": "Loan approval prediction based on applicant details"
    }
    
    with open(DATA_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


if __name__ == "__main__":
    df, data_path = download_or_load_dataset()
    print(f"Loaded dataset from {data_path}, shape: {df.shape}")
    validation_report = validate_dataset(df)
    print(f"Validation: {validation_report}")
    save_dataset_info(df, validation_report)
    perform_eda(df)
    print("Data preparation complete")
