import pandas as pd

from src.modelling.model.model import Model
from src.utils.mlflow import mlflow_track_evaluation


@mlflow_track_evaluation
def predict(model: Model, X: pd.DataFrame) -> pd.DataFrame:
    """Prediction function that calls predict method of model.

    Returns:
        pd.DataFrame: _description_
    """
    return model.predict(X)
