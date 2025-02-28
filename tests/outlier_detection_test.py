from src.outlier_detection import (
    OutlierDetector,
    ZScoreMethod,
    IQRMethod,
)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_outlier_detector(strategy: str) -> OutlierDetector:
    if strategy == 'zscore':
        return OutlierDetector(ZScoreMethod())
    elif strategy == 'iqr':
        return OutlierDetector(IQRMethod())
    else:
        logger.error(f"No matched strategy '{strategy}' exists.")
        raise ValueError(f"Provide a valid strategy (zscore, iqr). Provided: {strategy}")


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
    logger.info(df_numeric.head())

    if strategy == 'zscore':
        outlier_detector = OutlierDetector(ZScoreMethod())
    elif strategy == 'iqr':
        outlier_detector = OutlierDetector(IQRMethod())
    else:
        logger.error(f"No matched strategy '{strategy}' exists.")
        raise ValueError(f"Provide a valid strategy (zscore, iqr). Provided: {strategy}")

    # Detect outliers
    # outlier_detector = get_outlier_detector(strategy)
    logger.info(f"outlier_detector: {outlier_detector}")
    
    outliers = outlier_detector.detect_outliers(df_numeric)
    logger.info(f"outliers: {outliers}")

    logger.info(f"Detected Outliers:\n{outliers}")

    # Handle outliers
    if handle_method not in ['remove', 'cap']:
        logger.error(f"Unsupported outlier handling method '{handle_method}'")
        raise ValueError(f"Handle method should be either 'remove' or 'cap'. Provided: {handle_method}")

    df_cleaned = outlier_detector.handle_outliers(df_numeric, method=handle_method)

    return df_cleaned



# def outlier_detection_step2(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
#     """Detects and removes outliers using OutlierDetector."""
#     logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

#     if df is None:
#         logging.error("Received a NoneType DataFrame.")
#         raise ValueError("Input df must be a non-null pandas DataFrame.")

#     if not isinstance(df, pd.DataFrame):
#         logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
#         raise ValueError("Input df must be a pandas DataFrame.")

#     if column_name not in df.columns:
#         logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
#         raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
#         # Ensure only numeric columns are passed
#     df_numeric = df.select_dtypes(include=[int, float])

#     outlier_detector = OutlierDetector(ZScoreMethod(threshold=3))
#     outliers = outlier_detector.detect_outliers(df_numeric)
#     df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
#     return df_cleaned

if __name__ == "__main__":
    data = pd.read_csv('./data/raw/train.csv')
    # logger.info(data.head())
    # logger.info(X_train)
    df_numeric = data.select_dtypes(include=[int, float])
    # outliers = detect_outliers(df_numeric)
    df_cleaned = outlier_detection_step(df_numeric, 'SalePrice')
    # df_cleaned = outlier_detection_step(df_numeric, 'SalePrice', handle_method='remove')
    logger.info(df_cleaned.head())