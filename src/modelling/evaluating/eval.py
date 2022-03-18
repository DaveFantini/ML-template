import pandas as pd
from src.modelling.model.model import Model


def evaluate(
    model: Model, prediction: pd.DataFrame, actual: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate model performance given ground truth.

    Args:
        model (Model): Trained model
        prediction (pd.DataFrame): pandas dataframe of shape (n_samples)
        containing model predicitons
        actual (pd.DataFrame): pandas dataframe of shape (n_samples, n_features)
        containing test data
    """
    pass
