import json
import requests
import pandas as pd

# URL of the mlflow prediction server
url = 'http://127.0.0.1:8000/invocations'

# sample data for prediction
data = {
    "dataframe_records": [
        {
            "Id": 7,
            "MSSubClass": 20,
            "MSZoning": "RL",
            "LotFrontage": 75,
            "LotArea": 10084,
            "Street": "Pave",
            "Alley": "NA",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "Neighborhood": "Somerst",
            "Condition1": "Norm",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "1Story",
            "OverallQual": 8,
            "OverallCond": 5,
            "YearBuilt": 2004,
            "YearRemodAdd": 2005,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "Stone",
            "MasVnrArea": 186,
            "ExterQual": "Gd",
            "ExterCond": "TA",
            "Foundation": "PConc",
            "BsmtQual": "Ex",
            "BsmtCond": "TA",
            "BsmtExposure": "Av",
            "BsmtFinType1": "GLQ",
            "BsmtFinSF1": 1369,
            "BsmtFinType2": "Unf",
            "BsmtFinSF2": 0,
            "BsmtUnfSF": 317,
            "TotalBsmtSF": 1686,
            "Heating": "GasA",
            "HeatingQC": "Ex",
            "CentralAir": "Y",
            "Electrical": "SBrkr",
            "1stFlrSF": 1694,
            "2ndFlrSF": 0,
            "LowQualFinSF": 0,
            "GrLivArea": 1694,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "FullBath": 2,
            "HalfBath": 0,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "KitchenQual": "Gd",
            "TotRmsAbvGrd": 7,
            "Functional": "Typ",
            "Fireplaces": 1,
            "FireplaceQu": "Gd",
            "GarageType": "Attchd",
            "GarageYrBlt": 2004,
            "GarageFinish": "RFn",
            "GarageCars": 2,
            "GarageArea": 636,
            "GarageQual": "TA",
            "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": 255,
            "OpenPorchSF": 57,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "PoolQC": "NA",
            "Fence": "NA",
            "MiscFeature": "NA",
            "MiscVal": 0,
            "MoSold": 8,
            "YrSold": 2007,
            "SaleType": "WD",
            "SaleCondition": "Normal"
        }
    ]
}

# convert to json format
json_data = json.dumps(data)

# set headers for request
headers = {"Content-Type": "application/json"}

# request prediction - sent POST request to the server
response = requests.post(url, data=json_data, headers=headers)

# check response code

if response.status_code == 200:
    prediction = response.json()
    print("Prediction: ", prediction)
else:
    print(f"Error: {response.status_code}")
    print(response.text)