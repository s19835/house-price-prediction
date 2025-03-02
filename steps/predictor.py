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
    # Start the service
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop('columns', None)  # Remove column if it's present
    data.pop('index', None)  # Remove index if it's present

    # Define the expected columns
    expected_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction