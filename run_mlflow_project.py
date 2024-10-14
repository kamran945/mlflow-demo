import mlflow.projects

# Define the project directory (local directory containing the MLproject file)
project_uri = "."  # R. as the MLproject file is in current working directory
experiment_name = "mlflow_test_MLproject"
entry_point = "log_regression"
parameters = {
    "inverse_of_regularization_strength": 1.0,
    "max_iter": 100,
    "random_state": 42,
}

# Run the MLflow project
mlflow.projects.run(
    uri=project_uri,
    experiment_name=experiment_name,
    entry_point=entry_point,
    parameters=parameters,
)
