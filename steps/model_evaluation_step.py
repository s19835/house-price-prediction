from src.model_evaluation import (
    ModelEvaluator,
    RegressionModelEvaluation
)
import pandas as pd
from typing import Tuple

from sklearn.pipeline import Pipeline
from zenml import step
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def model_evaluation_step(trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, float]:
    '''
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    '''
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test should be a pandas data frame.")
    if not isinstance(y_test, pd.Series):
        raise ValueError("y_test should be a pandas series")
    
    logger.info("Applying the same preprocessing to the test data.")

    # apply preprocessing to test data
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # use regression strategy
    evaluator = ModelEvaluator(RegressionModelEvaluation())

    evaluate_matrics = evaluator.model_evaluate(
        trained_model.named_steps['model'], X_test_processed, y_test
    )

    if not isinstance(evaluate_matrics, dict):
        raise ValueError("Evaluation matrics must be returned as dictionary")
    
    mse = evaluate_matrics.get("Mean squred error", None)

    return evaluate_matrics, mse