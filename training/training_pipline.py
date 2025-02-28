from zenml import Model, pipeline
from steps.load_data_step import data_load_step
from steps.missing_value_handling_step import missing_value_handling_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step

@pipeline(
    model=Model(
        name="price-prediction"
    )
)

def ml_pipline():
    '''Define an end to end ML pipline'''

    # load data step
    data = data_load_step(
        "./data/raw/test.csv"
    )

    # handling missing values
    handled_data = missing_value_handling_step(data)

    # feature engineering
    engineered_data = feature_engineering_step(
        handled_data, strategy='log', features=['GrLivArea', 'SalePrice']
    )

    # outlier detection
    clearned_data = outlier_detection_step(engineered_data, column_name='SalePrice')