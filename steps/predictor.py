from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

import numpy as np
import pandas as pd
import json

@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    '''
    Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    '''
    # start the service
    service.start(timeout=10)

    # load the input data from json str
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop('columns', None)  # remove column if it's present
    data.pop('index', None)  # remove index if it's present

    # define the expected columns
    expected_columns = [
        "Id", "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
        "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
        "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars",
        "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
        "PoolArea", "MiscVal", "MoSold", "YrSold"
    ]

    # Ensure the input data has the correct number of columns
    if len(data['data'][0]) != len(expected_columns):
        raise ValueError(f"Expected {len(expected_columns)} columns, but got {len(data['data'][0])} columns")

    df = pd.DataFrame(data['data'], columns=expected_columns)

    # convert the data frame to json for prediction
    data_json = df.to_json(orient='split')

    # Run the prediction
    prediction = service.predict(data_json)

    return prediction



