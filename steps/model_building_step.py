from typing import Annotated
import pandas as pd
from zenml import step, Model, ArtifactConfig
from zenml.client import Client
from zenml.enums import ArtifactType

import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# get active experiment tracker from zenml
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name='price-prediction',
    version='0.1.0',
    license='Apache 2.0',
    description='prediction model for house prices'
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[Pipeline, ArtifactConfig(name='sklearn-pipline', artifact_type=ArtifactType.MODEL)]:
    '''
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    '''

    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("input X_train must be a pandas data frame")
    if not isinstance(y_train, pd.Series):
        raise ValueError("input y_train must be a pandas Series")
    
    # identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include='number').columns

    # if numerical_cols == X_train.select_dtypes(exclude=['object', 'category']):
    #     logger.warning("unexpected data type present in the data frame")
    
    logger.info(f"Categorical Columns: {categorical_cols.tolist()}")
    logger.info(f"Numerical Columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformar = SimpleImputer(strategy='mean')
    categorical_transformar = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        [
            ('num', numerical_transformar, numerical_cols),
            ('cat', categorical_transformar, categorical_cols),
        ]
    )

    # Defineing model training
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    # start mlflow to log model process
    if not mlflow.active_run():
        mlflow.start_run()
    
    try:
        # autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logger.info("Building and Training Linear Regression Model...")
        pipeline.fit(X_train, y_train)

        # log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        )

        onehot_encoder.fit(X_train[categorical_cols])
        expected_cols = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logger.info(f"Model expects the following columns: {expected_cols}")

    except Exception as e:
        logger.error(f"Error during model trainning: {e}")
        raise e
    
    finally:
        mlflow.end_run()
    
    return pipeline