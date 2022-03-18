import pandas as pd

from src.modelling.model.model import Model
from src.utils.mlflow import mlflow_track_training


@mlflow_track_training
def train(X: pd.DataFrame, y: pd.DataFrame) -> Model:
    """Training function that initialize and fit the model.
       Multiple additional parameters are possible to customize training.

    Args:
        X (pd.DataFrame): _description_
        y (pd.DataFrame): _description_

    Returns:
        Model: Trained model
    """
    model = Model().fit(X, y)
    return model
