from calendar import c
from pathlib import Path
import tempfile
import mlflow
import pandas as pd
from src.modelling.evaluating.eval import evaluate
from src.modelling.model.model import Model
from src.utils.config import Config
from src.utils.utils import create_artifact


class MlFlowModel(mlflow.pyfunc.PythonModel):
    """
    Wrapping class to save model with MlFlow
    """

    def __init__(self, model: Model, parameters: dict):
        self.model = model
        self.parameters = parameters

    def predict(self, context, X: pd.DataFrame) -> pd.DataFrame:
        # il context serve a MLflow, non rimuoverlo
        return self.model.predict(X)

    def get_parameters(self) -> dict:
        return self.parameters


def mlflow_track_training(training_function):
    """
    Decorator function that add MLflow logging to model training
    """

    def wrapper(*args, **kwargs):
        # TODO LOG instead of print
        print("Decorating with mlflow experiment")
        mlflow_enabled = True
        model = training_function(*args, **kwargs)

        if mlflow_enabled:
            model_name = "Model"
            set_mlflow_params("file:/Users/david/Desktop/ML-template/experiments/", "Project")
            with mlflow.start_run(nested=True, run_name=model_name) as run:
                # Add experiment ID to model to be used in evaluation
                model.set_mlflow_run_id(run.info.run_id)
                # Log model
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=MlFlowModel(model, model.params),
                )
                # Log parameters
                test_param = kwargs.get("test_param", "")
                mlflow.log_params({"test_param": test_param})
        return model

    return wrapper


def mlflow_track_evaluation(predict_function):
    """
    Decorator function that add MLflow logging to model forecasting/evaluation
    """

    def wrapper(*args, **kwargs):
        mlflow_enabled = True
        forecast = predict_function(*args, **kwargs)

        if mlflow_enabled:
            set_mlflow_params("file:/Users/david/Desktop/ML-template/experiments/", "Project")
            model = kwargs.get("model")
            actual = pd.DataFrame()
            run_id = model.get_mlflow_run_id()
            with mlflow.start_run(run_id=run_id):
                evaluation = evaluate(model, forecast, actual)
                # Log performance metrics
                mlflow.log_metrics(evaluation["numeric_metrics"])
                save_artifacts(evaluation)

        return forecast

    return wrapper


def set_mlflow_params(model_output_path: str, project_name: str) -> None:
    """Set up mlflow run parameters.

    Args:
        model_output_path (str): _description_
        project_name (str): _description_
    """
    config = Config()
    experiments_path = Path(config.mlflow["model_output_path"]) / "mlruns"
    project_name = config.mlflow["project_name"]
    mlflow.set_tracking_uri(experiments_path.as_posix())
    mlflow.set_experiment(project_name)


def save_artifacts(artifacts: dict) -> None:
    """Save artifacts (plots and csv) to file

    Args:
        artifacts (dict): _description_
    """
    with tempfile.TemporaryDirectory() as tmp_directory:
        csvs = artifacts["csvs"]
        for name in csvs:
            mlflow.log_artifact(
                create_artifact(csvs[name].encode("utf-8"), tmp_directory,
                                name, "csv"),
                artifact_path="csv",
            )
        plots = artifacts["plots_png"]
        for name in plots:
            mlflow.log_artifact(
                create_artifact(plots[name], tmp_directory, name, "png"),
                artifact_path="plots",
            )
        plots = artifacts["plots_html"]
        for name in plots:
            mlflow.log_artifact(
                create_artifact(plots[name], tmp_directory, name, "html"),
                artifact_path="plots",
            )
