from src.data_splitting import (
    DataSplitter,
    TrainTestSplit,
    KFoldSplit,
    StratifiedSplit
)
from typing import Tuple
import pandas as pd
from zenml import step

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class DataSplittingFactory:
    @staticmethod
    def get_data_splitter(strategy: str) -> DataSplitter:
        if strategy == 'train_test_split':
            data_splitter = DataSplitter(TrainTestSplit())
        elif strategy == 'kfold_split':
            data_splitter = DataSplitter(KFoldSplit())
        elif strategy == 'stratified_split':
            data_splitter = DataSplitter(StratifiedSplit())
        else:
            logger.error(f"No matched strategy '{strategy}' exists.")
            raise ValueError(f"Provide a valid strategy (train_test_split, kfold_split, stratified_split). Provided: {strategy}")
        
        return data_splitter


@step
def data_splitting_step(df: pd.DataFrame, target_column: str, method: str='train_test')-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if df is None:
        logger.error("Received a NoneType DataFrame")
        raise ValueError("Input df must be non-null")
        
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame")
        
    if target_column not in df.columns:
        logger.error(f"Column '{target_column}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    
    if method == 'kfold_split':
        logger.info(f"Unsupported method {method}: Method is not yet fully functional")
        return None
    
    data_splitter = DataSplittingFactory.get_data_splitter(strategy=method)
    
    return data_splitter.split_data(df, target_column)
    
    