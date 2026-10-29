#!/usr/bin/env python
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: {description} returned non-zero exit code")
    return result.returncode

def main():
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("Starting Loan Approval Prediction Pipeline")
    
    # Setup dependencies
    run_command("py -m pip install -r requirements.txt", "Installing dependencies")
    
    # Data preparation
    run_command("py src/data_preparation.py", "Data preparation and EDA")
    
    # Training
    run_command("py src/train.py", "Model training")
    
    # Tests
    run_command("py -m pytest tests/ -v", "Running tests")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)

if __name__ == "__main__":
    main()
