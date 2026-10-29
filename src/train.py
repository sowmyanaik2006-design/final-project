import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from data_preparation import (
    download_or_load_dataset,
    preprocess_data,
    DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR
)

RANDOM_STATE = 42


def train_models(X_train, y_train, preprocessor):
    """Train baseline and candidate models."""
    models = {}
    
    # Baseline model - Logistic Regression
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    models["logistic_regression"] = lr_pipeline
    
    # Random Forest
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ])
    rf_pipeline.fit(X_train, y_train)
    models["random_forest"] = rf_pipeline
    
    # Gradient Boosting
    gb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ])
    gb_pipeline.fit(X_train, y_train)
    models["gradient_boosting"] = gb_pipeline
    
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate a model on test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    
    return metrics, y_pred


def save_model_artifacts(model, model_name, metrics):
    """Save model and metadata."""
    model_path = MODELS_DIR / "loan_approval_model.pkl"
    joblib.dump(model, model_path)
    
    metadata = {
        "model_name": model_name,
        "model_type": type(model.named_steps["classifier"]).__name__,
        "saved_at": datetime.now().isoformat(),
        "metrics": metrics,
        "random_state": RANDOM_STATE
    }
    
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model_path


if __name__ == "__main__":
    df, _ = download_or_load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features = preprocess_data(df)
    
    print("Training models...")
    models = train_models(X_train, y_train, preprocessor)
    
    print("\nEvaluating models on test set:")
    all_results = {}
    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        all_results[name] = metrics
        print(f"{name}: {metrics}")
    
    # Select best model based on F1 score
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]["f1_score"])
    best_model = models[best_model_name]
    best_metrics = all_results[best_model_name]
    
    print(f"\nBest model: {best_model_name} with F1={best_metrics['f1_score']:.4f}")
    
    # Save best model
    model_path = save_model_artifacts(best_model, best_model_name, best_metrics)
    print(f"Model saved to {model_path}")
    
    # Save all results
    with open(METRICS_DIR / "training_results.json", "w") as f:
        json.dump({
            "all_results": all_results,
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print("Training complete")
