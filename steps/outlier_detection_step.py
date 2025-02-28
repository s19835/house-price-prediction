from src.outlier_detection import (
    OutlierDetector,
    ZScoreMethod,
    IQRMethod,
)
import pandas as pd
from zenml import step

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class OutlierDetectionFactory:
    @staticmethod
    def get_outlier_detector(strategy: str) -> OutlierDetector:
        if strategy == 'zscore':
            return OutlierDetector(ZScoreMethod())
        elif strategy == 'iqr':
            return OutlierDetector(IQRMethod())
        else:
            logger.error(f"No matched strategy '{strategy}' exists.")
            raise ValueError(f"Provide a valid strategy (zscore, iqr). Provided: {strategy}")

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str, strategy: str = 'zscore', handle_method: str = 'remove') -> pd.DataFrame:
    '''Detects and removes outliers using outlier detection strategies'''
    if df is None:
        logger.error("Received a NoneType DataFrame")
        raise ValueError("Input df must be non-null")
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame")
    
    if column_name not in df.columns:
        logger.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    df_numeric = df.select_dtypes(include=[int, float])

    # Detect outliers
    outlier_detector = OutlierDetectionFactory.get_outlier_detector(strategy)
    outliers = outlier_detector.detect_outliers(df_numeric)
    logger.info(f"Detected Outliers:\n{outliers}")
    
    # Handle outliers
    if handle_method not in ['remove', 'cap']:
        logger.error(f"Unsupported outlier handling method '{handle_method}'")
        raise ValueError(f"Handle method should be either 'remove' or 'cap'. Provided: {handle_method}")
    
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method=handle_method)
    
    return df_cleaned