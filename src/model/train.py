# Import libraries
import argparse
import glob
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define functions
def main(args):
    # Enable autologging
    mlflow.sklearn.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df, test_size=0.2, random_state=42):
    """Splits the DataFrame into training and testing sets."""
    logging.info("Splitting data into training and testing sets.")
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    logging.info(f"Training model with regularization rate: {reg_rate}")
    # Train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    # Log model
    mlflow.log_param("regularization_rate", reg_rate)

def parse_args():
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data', type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args

# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")
