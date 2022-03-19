"""Test implemented in pytest format. (https://docs.pytest.org/en/7.0.x/)

   pytest discovers all tests following its Conventions,
   so it finds test_ prefixed functions and Test prefixed classes.

   To run tests, in project root
   ::>>> pytest
"""
import numpy as np
import pandas as pd
import os
from src.__main__ import main
from src.modelling.model.model import Model
from src.modelling.training.training import train
from src.utils.constants import OP_DATA_GENERATION


def test_main():
    """Test main entrypoint.
    """
    main(OP_DATA_GENERATION)
    assert True

def generate_sample_input(array=False):        
    X = pd.DataFrame({"x_0": [1, 1, 2, 3], "x_1": [1, 2, 2, 3]})
    # y = 1 * x_0 + 2 * x_1 + 3
    X_array = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_array = np.dot(X_array, np.array([1, 2])) + 3
    y = pd.DataFrame({"y": y_array})
    return X, y

class TestModel:
    """Class to group tests on Model class
    """
    def test_model(self):
        """Test model training with sample data
        """
        X, y = generate_sample_input()
        m = Model().fit(X, y)
        assert m.model is not None

    
class TestMLflow:
    """Class to group tests on MLflow
    """
    def test_mlflow_training(self):
        """Test mlflow training and model save
        """
        X, y = generate_sample_input()
        model = train(X, y)
        # TODO check params
        assert len(os.listdir("./experiments/")) > 0
