from zenml import Model, pipeline
from steps.load_data_step import data_load_step
from steps.missing_value_handling_step import missing_value_handling_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitting_step import data_splitting_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step

@pipeline(
    model=Model(
        name="price-prediction"
    )
)

def ml_pipeline():
    '''Define an end to end ML pipeline'''

    # load data step
    data = data_load_step(
        "./data/raw/train.csv"
    )

    # handling missing values
    handled_data = missing_value_handling_step(data)

    # feature engineering
    engineered_data = feature_engineering_step(
        handled_data, strategy='log', features=['GrLivArea', 'SalePrice']
    )

    # outlier detection
    clearned_data = outlier_detection_step(engineered_data, column_name='SalePrice')

    # split data
    X_train, X_test, y_train, y_test = data_splitting_step(clearned_data, target_column="SalePrice")

    # model building
    model = model_building_step(X_train, y_train)

    # evaluate the model
    evaluation_matrics, mse = model_evaluation_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model

if __name__ == "__main__":
    run = ml_pipeline
