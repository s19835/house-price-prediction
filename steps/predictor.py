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
    data.pop('columns', None) # rmv col if its present
    data.pop('index', None) # rmv index if its present

    # define the expected columns
    expected_columns = [
        "Id", "MSSubClass", "LotFrontage", "LotArea", "YearBuilt", "YearRemod/Add",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
        "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
        "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
        "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "ScreenPorch",
        "PoolArea", "MiscVal", "MoSold", "YrSold"
    ]

    df = pd.DataFrame(data['data'], columns=expected_columns)

    # convert the data frame to json for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction



