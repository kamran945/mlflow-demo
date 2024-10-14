# mlflow-demo

mlflow demo

# MLFlow:

- MLflow is an open-source platform designed to manage the machine learning lifecycle.
- It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

## Repository Description:

- This repository contains a collection of Python scripts demonstrating the use of different MLflow features as learned from Course **"MLflow in Action - Master the art of MLOps using MLflow tool"**, Link: https://www.udemy.com/course/mlflow-course
- Each script is structured to provide hands-on examples for specific features.
- This repository contains different scripts implementing different MLflow features inclduing:
  - Experiment Tracking: Logging metrics, parameters, and artifacts to track experiments.
  - Autologging: Automatically logging machine learning models and their metrics.
  - MLflow Projects: Managing machine learning projects with defined environments and parameters.
  - Custom Metrics and Evaluation: Implementing custom metrics and validation using mlflow.evaluate.
  - Model Registry: Saving and registering models for reproducibility and deployment.
  - MLflow Tracking Server: Setting up and using an MLflow tracking server for remote logging.
  - MLflow Client: Setting up experiments using MLflow client.

## Instructions to Use:

- **Install Packages**:

  - Install Required Packages (if not already installed):
    - e.g.: pip install scikit-learn
  - Setting up Mlflow Tracking Server:
    - **mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000**
  - Run the Script:
    - **python script_name.py --C <inverse_of_regularization_strength> --max_iter <max_iterations> --random_state <random_state>**
    - **Note**: Replace script_name.py with the desired filename.
  - Example command:
    **python script_name.py --C 0.5 --max_iter 200 --random_state 42**

  - Output:
    - The script will display the evaluation metrics and a classification report for the model's performance on the test set.
    - Tracking URI has been set to 127.0.0.1:5000. Results can be found on http://127.0.0.1:5000

## MLproject:

- MLproject is a file format used by MLflow to define a machine learning project.
- The MLproject file helps standardize the process by defining:

  - Environment: Specifies dependencies (e.g., Python packages) required to run the project.
  - Parameters: Defines input parameters for model training (like hyperparameters).
  - Entry Points: Designates scripts or commands that can be executed (e.g., training a model or evaluating results).

  - Running MLproject:
    - From Python Script:
      - **python run_mlflow_project.py**
    - Using CLI:
      - **mlflow run --entry-point log_regression --experiment-name "mlflow_test_MLproject" .**
