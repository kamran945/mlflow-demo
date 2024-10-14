import argparse
import warnings
import logging
import sys
import os

from pathlib import Path
import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.dummy import DummyClassifier

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from mlflow.models.evaluation import MetricThreshold
from mlflow.metrics import make_metric
from mlflow import MlflowClient

import joblib
import cloudpickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def save_data(df, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def get_data(data_filepath):
    if not data_filepath.is_file():
        logger.info("wine dataset not found. Creating a new one...")
        data = load_wine()
        # Convert to a DataFrame
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        df["target"] = data.target
        save_data(df, data_filepath)
    else:
        logger.info("wine dataset found. Loading it...")
        df = pd.read_csv(data_filepath)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Classifier for wine Dataset"
    )

    # Add arguments for Logistic Regression hyperparameters
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations (default: 100)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )

    return parser.parse_args()


# Function to compute evaluation metrics
def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="weighted"
    )  # average='weighted' for multi-class
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    return accuracy, precision, recall, f1


def main():
    warnings.filterwarnings("ignore")

    # Parse the command-line arguments
    args = parse_args()

    np.random.seed(args.random_state)

    # Load or create the wine dataset
    data_filepath = Path.cwd().joinpath("data", "data.csv")
    df = get_data(data_filepath)

    # Split the dataset into training and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=args.random_state)

    data_filepath = Path.cwd().joinpath("data", "train_data.csv")
    save_data(train, data_filepath)
    data_filepath = Path.cwd().joinpath("data", "test_data.csv")
    save_data(test, data_filepath)

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # Set experiment uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    print("Experiment uri set to: ", mlflow.get_tracking_uri())

    # Initialize MLflow client
    client = MlflowClient()

    # Create an experiment
    experiment_name = "Mlflow Client Experiment"
    try:
        experiment_id = client.create_experiment(experiment_name)
    except Exception as e:
        # If experiment already exists, get its ID
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    exp = client.get_experiment_by_name(experiment_name)

    print("Name: {}".format(exp.name))
    print("Experiment ID: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Lifecycle State: {}".format(exp.lifecycle_stage))
    print("Tags: {}".format(exp.tags))
    print("Creation Time: {}".format(exp.creation_time))

    # Start a run under the created experiment
    run = client.create_run(experiment_id)
    run_id = run.info.run_id

    # Train the model
    model = LogisticRegression(
        C=args.C, max_iter=args.max_iter, random_state=args.random_state
    )
    model.fit(X_train, y_train)

    # Log model parameters using mlflow client
    client.log_param(run_id=run_id, key="C", value=args.C)
    client.log_param(run_id=run_id, key="max_iter", value=args.max_iter)
    client.log_param(run_id=run_id, key="random_state", value=args.random_state)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy, precision, recall, f1 = eval_metrics(
        y_true=y_test,
        y_pred=y_pred,
    )
    # Log metrics using mlflow client
    client.log_metric(
        run_id=run_id,
        key="accuracy",
        value=accuracy,
        step=1,
    )
    client.log_metric(
        run_id=run_id,
        key="precision",
        value=precision,
        step=1,
    )
    client.log_metric(
        run_id=run_id,
        key="recall",
        value=recall,
        step=1,
    )
    client.log_metric(
        run_id=run_id,
        key="f1_score",
        value=f1,
        step=1,
    )

    # Save the model and log it as an artifact
    model_dir = "client_model"
    os.makedirs(model_dir, exist_ok=True)
    mlflow.sklearn.save_model(model, model_dir)
    client.log_artifact(run_id, model_dir)

    # End the run (mark it as successful)
    client.set_terminated(run_id, status="FINISHED")

    # Register the model to the model registry
    model_name = "log_reg_model_client"
    model_uri = f"runs:/{run_id}/{model_dir}"

    try:
        # Try to register a new model
        client.create_registered_model(model_name)
    except Exception as e:
        # If the model already exists, pass
        pass

    # Create a new version of the registered model
    client.create_model_version(model_name, model_uri, run_id)

    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
