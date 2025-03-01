from abc import ABC, abstractmethod
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# define a base class (interface for strategy)
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        '''
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A *trained* scikit-learn model instance.
        '''
        pass

# concreate classes with specific strategies
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        '''
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        '''
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("input X_train should be a pandas data frame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("input y_train should be a pandas series")
        
        logger.info("Initializing linear regression model with scaler...")

        # pipline with standard scaling and linear regression
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('model', LinearRegression()),
            ]
        )

        # train the model
        logger.info("Trainning the model...")
        pipeline.fit(X=X_train, y=y_train)

        return pipeline

# context class for strategies
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        '''
        Initialize the model builder with specific strategy

        parameters: 
        strategy (ModelBuildingStrategy): strategy to use for model building (eg: linear_regression)
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        '''
        Set new strategy for model building and training

        parameters: 
        strategy (ModelBuildingStrategy): new strategy to use for model building
        '''
        self.strategy = strategy
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        '''
        build and train the model using specific strategy (linear_regression)

        parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        return:
        RegressorMixin: A trained scikit-learn model instance.
        '''
        return self.strategy.build_and_train_model(X_train, y_train)