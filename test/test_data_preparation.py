import os
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preparation import download_or_load_dataset, validate_dataset


def test_dataset_loads():
    df, path = download_or_load_dataset()
    assert df is not None
    assert len(df) > 0
    assert os.path.exists(path)


def test_dataset_validation():
    df, _ = download_or_load_dataset()
    validation_report = validate_dataset(df)
    assert validation_report["shape"][0] > 0
    assert "Loan_Status" in df.columns
    assert validation_report["duplicates"] >= 0


def test_target_exists():
    df, _ = download_or_load_dataset()
    assert "Loan_Status" in df.columns
    assert df["Loan_Status"].isin(["Y", "N"]).all()
