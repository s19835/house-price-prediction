from abc import ABC, abstractmethod
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# define strategy interface
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def model_evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        '''
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        '''
        pass

# concreate class with specific strategy
class RegressionModelEvaluation:
    def model_evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        '''
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        '''
        logger.info("Predict using the model...")
        y_pred = model.predict(X_test)

        # evaluate model
        logger.info("Calcualte the evaluation metrics...")
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)

        metrice = {"Mean squred error": mse, "R-Squred": r2}
        logger.info(f"Model Evaluation Matrics: {metrice}")

        return metrice

# context class
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        '''
        initialize model evaluation with specific strategy

        parameters:
        strategy (ModelEvaluationStrategy): strategy to use to evaluate model
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelEvaluationStrategy):
        '''
        set new strategy to use to evaluate model

        parameters:
        strategy (ModelEvaluationStrategy): new strategy to evaluate the model
        '''
        self.strategy = strategy

    def model_evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        '''
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        '''
        logger.info("Evaluate model using speicfied strategy...")
        return self.strategy.model_evaluate(model, X_test, y_test)