import argparse
import warnings
import logging
import sys

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

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import mlflow.pyfunc

import joblib
import cloudpickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def save_data(df, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def get_data(data_filepath):
    if not data_filepath.is_file():
        logger.info("Wine dataset not found. Creating a new one...")
        wine_data = load_wine()
        # Convert to a DataFrame
        df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
        df["target"] = wine_data.target
        save_data(df, data_filepath)
    else:
        logger.info("Wine dataset found. Loading it...")
        df = pd.read_csv(data_filepath)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Classifier for Wine Dataset"
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
    data_filepath = Path.cwd().joinpath("data", "wine_data.csv")
    df = get_data(data_filepath)

    # Split the dataset into training and test sets
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=args.random_state,
    )

    data_filepath = Path.cwd().joinpath("data", "train_wine_data.csv")
    save_data(train, data_filepath)
    data_filepath = Path.cwd().joinpath("data", "test_wine_data.csv")
    save_data(test, data_filepath)

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # Set experiment uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    print("Experiment uri set to: ", mlflow.get_tracking_uri())

    # Create Experiment using mlflow
    # exp_id = mlflow.create_experiment(
    #     name="mlflow_custom_model", tags={"version": "v1.0"}
    # )
    # exp = mlflow.get_experiment(exp_id)
    exp = mlflow.set_experiment(experiment_name="mlflow_custom_model")
    exp_id = exp.experiment_id

    print("Name: {}".format(exp.name))
    print("Experiment ID: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Lifecycle State: {}".format(exp.lifecycle_stage))
    print("Tags: {}".format(exp.tags))
    print("Creation Time: {}".format(exp.creation_time))

    with mlflow.start_run(experiment_id=exp_id):
        # enable auto logging for mllow experiment
        # Automatically log hyperparameters, metrics, and artifacts from MLflow
        # This will make it easier to visualize and compare the results of different experiments in the MLflow UI
        # and in the MLflow experiment tracking server
        # disable log_model, log_input_examples and log_signatures (this is done to manually log them later on)
        # mlflow.autolog(
        #     log_models=False,
        #     log_input_examples=False,
        #     log_model_signatures=False,
        # )

        # Create and train the Logistic Regression model using inputs from the command line
        model = LogisticRegression(
            C=args.C, max_iter=args.max_iter, random_state=args.random_state
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy, precision, recall, f1 = eval_metrics(
            y_true=y_test,
            y_pred=y_pred,
        )

        mlflow.set_tag("run", "1")

        # infer model signature
        signature = infer_signature(model_input=X_test, model_output=y_test)

        # get input examples
        input_examples = {
            "columns": np.array(X_test.columns),
            "data": np.array(X_test.values),
        }

        sklearn_model_path = Path.cwd() / "sklearn_model_store/sklearn_model.pkl"
        sklearn_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, sklearn_model_path)
        artifacts = {"sklearn_model": str(sklearn_model_path), "data": "data/"}

        PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        conda_env = {
            "channels": ["defaults"],
            "dependencies": [
                f"python={PYTHON_VERSION}",
                "pip",
                {
                    "pip": [
                        f"mlflow=={mlflow.__version__}",
                        f"sklearn=={sklearn.__version__}",
                        f"cloudpickle=={cloudpickle.__version__}",
                    ],
                },
            ],
            "name": "sklearn_conda_env",
        }

        class SklearnWrapperClass(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

            def predict(self, context, model_input, params=None):
                return self.sklearn_model.predict(model_input.values)

        mlflow.pyfunc.log_model(
            artifact_path="sklearn_model_pyfunc",
            python_model=SklearnWrapperClass(),
            conda_env=conda_env,
            artifacts=artifacts,
            code_path=["main.py"],
            signature=signature,
            input_example=input_examples,
        )
        # mlflow.sklearn.log_model(
        #     model,
        #     "log_reg_model_1",
        #     signature=signature,
        #     input_example=input_examples,
        # )


if __name__ == "__main__":
    main()
