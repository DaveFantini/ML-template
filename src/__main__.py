import argparse
from src.data.data_generation import get_test_data, get_train_data
from src.modelling.forecasting.forecasting import predict
from src.modelling.model.model import Model
from src.modelling.training.training import train

from src.utils.constants import (
    OP_DATA_GENERATION,
    OP_PREDICT,
    OP_TRAINING,
    OP_UPDATE_RESULT,
    OPERATION_LIST,
)


def main(operation):
    if operation == OP_DATA_GENERATION:
        # Prepare training data
        get_train_data("2021-01-01", "2022-01-01")
        # Prepare test data
        get_test_data("2022-01-01", "2023-01-01")
        print("Data Generation")

    elif operation == OP_TRAINING:
        # Train and save model
        df_train = get_train_data("2021-01-01", "2022-01-01")
        model = train(df_train, df_train)
        model.save_model()
        print("Model Training")

    elif operation == OP_PREDICT:
        # Forecast and save results
        df_test = get_test_data("2022-01-01", "2023-01-01")
        model = Model().load_model()
        predict(model, df_test)
        print("Model Forecast")

    elif operation == OP_UPDATE_RESULT:
        # Postprocess resutls
        print("Process results")

    else:
        raise ValueError("Variable operation wrongly valorized")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--operation",
        help="Operation to execute",
        choices=OPERATION_LIST,
        dest="operation",
        required=True,
    )

    args = parser.parse_args()
    main(operation=args.operation)
