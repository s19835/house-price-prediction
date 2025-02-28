# Unit tests for data preprocessing
from src.data_splitting import (
    DataSplitter,
    TrainTestSplit,
    StratifiedSplit
)

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_data_splitter(strategy: str) -> DataSplitter:
    if strategy == 'train_test_split':
        data_splitter = DataSplitter(TrainTestSplit())
    # elif strategy == 'kfold_split':
    #     data_splitter = DataSplitter(KFoldSplit())
    elif strategy == 'stratified_split':
        data_splitter = DataSplitter(StratifiedSplit())
    else:
        logger.error(f"No matched strategy '{strategy}' exists.")
        raise ValueError(f"Provide a valid strategy (train_test_split, kfold_split, stratified_split). Provided: {strategy}")
    
    return data_splitter

def data_splitting_step(df: pd.DataFrame, target_column: str, method: str='train_test_split')-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if df is None:
        logger.error("Received a NoneType DataFrame")
        raise ValueError("Input df must be non-null")
        
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame")
        
    if target_column not in df.columns:
        logger.error(f"Column '{target_column}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    
    # if method == 'kfold_split':
    #     logger.info(f"Unsupported method {method}: Method is not yet fully functional")
    #     return None
    
    data_splitter = get_data_splitter(strategy=method)
    
    return data_splitter.split_data(df, target_column)

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Detect outliers using the Z-score method.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            pd.DataFrame: A boolean indicating outliers (True = outlier).
        '''
        logger.info("Detecting Outliers Using Z-Score method...")

        z_score = np.abs((df - df.mean()) / df.std())
        outliers = z_score > 3
        return outliers


def handle_outliers(df: pd.DataFrame, method='remove', **kwargs) -> pd.DataFrame:
    '''
    by using methods like remove, cap handle outliers

    parameters:
    df (pd.DataFrame): Data Frame that has outliers
    method (str): either remove or cap
    **kwargs (any): any other inputs

    return:
    pd.DataFrame: pandas data frame without outliers
    '''
    outliers = detect_outliers(df)

    if outliers is None:
        logger.error("Outlier detection returned None")
        raise ValueError("Outlier detection must return a DataFrame or Series")
    
    if method == 'remove':
        logger.info("Removing Outliers...")
        df_cleaned = df[(~outliers).all(axis=1)]
    
    elif method == 'cap':
        logger.info("Capping Outliers...")
        df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    
    else:
        logger.warning(f"Unknown method '{method}'. No outlier handling performed")
        return df
    
    return df_cleaned

if __name__ == "__main__":
    data = pd.read_csv('./data/raw/train.csv')
    # logger.info(data.head())
    X_train, X_test, y_train, y_test = data_splitting_step(data, 'SalePrice', 'train_test_split')
    # logger.info(X_train)
    df_numeric = data.select_dtypes(include=[int, float])
    # outliers = detect_outliers(df_numeric)
    # df_cleaned = handle_outliers(df_numeric, method='remove')
    # df_cleaned = outlier_detection_step(df_numeric, 'SalePrice', handle_method='remove')
    logger.info(df_cleaned.head())