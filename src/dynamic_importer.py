import pandas as pd
from zenml import step

@step
def dynamic_importer() -> str:
    ''' Dynamically import data for testing '''

    # api call, database query or loading from a file
    data = {
        "Order": [1, 2],
        "Id": [1461, 1462],
        "MSSubClass": [20, 20],
        "LotFrontage": [80.0, 81.0],
        "LotArea": [11622, 14267],
        "YearBuilt": [2006, 1926],
        "YearRemod/Add": [2007, 1961],
        "BsmtFinSF1": [1256, 920],
        "BsmtFinSF2": [0, 0],
        "BsmtUnfSF": [50, 406],
        "TotalBsmtSF": [1306, 1326],
        "1stFlrSF": [1306, 1326],
        "2ndFlrSF": [0, 0],
        "GrLivArea": [1306, 1326],
        "BsmtFullBath": [1, 1],
        "BsmtHalfBath": [0, 0],
        "FullBath": [2, 2],
        "HalfBath": [1, 0],
        "BedroomAbvGr": [2, 3],
        "KitchenAbvGr": [1, 1],
        "TotRmsAbvGrd": [5, 6],
        "Fireplaces": [1, 1],
        "GarageCars": [2, 2],
        "GarageArea": [576, 500],
        "WoodDeckSF": [349, 0],
        "OpenPorchSF": [0, 60],
        "ScreenPorch": [120, 0],
        "PoolArea": [0, 0],
        "MiscVal": [0, 12500],
        "MoSold": [6, 6],
        "YrSold": [2010, 2010],
    }

    df = pd.DataFrame(data)

    # dataframe to json string
    json_data = df.to_json(orient="split")

    return json_data
