import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class Model:
    def __init__(self, **kwargs) -> None:
        self.model = None
        self.params = kwargs.copy()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit model.

        Args:
            X (pd.DataFrame): pandas dataframe of shape (n_samples, n_features)
                              Training data.
            y (pd.DataFrame): pandas dataframe of shape (n_samples,) or
                              (n_samples, n_targets) Target values.

        Returns:
            self (object): fitted model.
        """
        # model = RandomForestRegressor(self.params)
        model = RandomForestRegressor()
        self.model = model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predicions with trained model.

        Args:
            X (pd.DataFrame): pandas dataframe of shape (n_samples, n_features).
                              Test data.

        Returns:
            pd.DataFrame: pandas dataframe of shape (n_samples).
        """
        return self.model.predict(X.values)

    def save_model(self):
        """Save model to file."""
        pass

    def load_model(self):
        """Load model from file."""
        pass
    
    def set_mlflow_run_id(self, run_id: str):
        self.params["run_id"] = run_id