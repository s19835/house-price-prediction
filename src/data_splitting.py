from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from typing import Tuple

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# base class for data splitting strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        '''
        Abstract class for splitting the data

        Parameters:
        df (pd.DataFrame): pandas data frame for splitting
        target_column (str): target column in the data set

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        '''
        pass

    def get_features_and_target(self, df: pd.DataFrame, target_column: str):
        '''
        Get features and target from the DataFrame.

        Parameters:
        df (pd.DataFrame): pandas data frame
        target_column (str): target column in the data set

        Returns:
        X, y: Features and target
        '''
        if df is None:
            logger.error("Received a NoneType DataFrame")
            raise ValueError("Input df must be non-null")
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected pandas DataFrame, got {type(df)} instead.")
            raise ValueError("Input df must be a pandas DataFrame")
        
        if target_column not in df.columns:
            logger.error(f"Column '{target_column}' does not exist in the DataFrame.")
            raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

# concrete classes

# train-test split
class TrainTestSplit(DataSplittingStrategy):
    def __init__(self, test_size: float=0.2, random_state: int=42):
        '''
        Initialize the strategy with test size and random state.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        '''
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
        Split the data into training and testing sets.

        Parameters:
            df (pd.DataFrame): The feature matrix.
            target_column (str): The target variable.

        Returns:
            X_train, X_test, y_train, y_test: Split data.
        '''
        logger.info("Performing train-test splitting...")

        X, y = self.get_features_and_target(df, target_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
    
# K-Fold
# class KFoldSplit(DataSplittingStrategy):
#     def __init__(self, n_splits:int=5, shuffle: bool=True, random_state: int=42):
#         '''
#         Initialize the strategy with number of splits, shuffle, and random state.

#         Parameters:
#             n_splits (int): Number of folds.
#             shuffle (bool): Whether to shuffle the data before splitting.
#             random_state (int): Random seed for reproducibility.
#         '''
#         self.n_splits = n_splits
#         self.shuffle = shuffle
#         self.random_state = random_state
    
#     def split_data(self, df: pd.DataFrame, target_column: str):
#         '''
#         Split the data into K folds for cross-validation.

#         Parameters:
#             df (pd.DataFrame): Data Frame that need to be splitted.
#             target_column (str): The target variable.

#         Returns:
#             KFold object: An iterator over the K folds.
#         '''
#         logger.info("Splitting data using K-Fold Cross-Validation...")

#         X, y = self.get_features_and_target(df, target_column)

#         return KFold(
#             n_splits=self.n_splits,
#             shuffle=self.shuffle,
#             random_state=self.random_state
#         ).split(X, y)
    
# stratified split
class StratifiedSplit(DataSplittingStrategy):
    def __init__(self, test_size: float=0.2, random_state: int=42):
        '''
        Initialize the strategy with test size and random state.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        '''
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
        Split the data into training and testing sets while preserving class distribution.

        Parameters:
           df (pd.DataFrame): Data Frame that need to be splitted.
           target_column (str): The target variable.
        Returns:
            X_train, X_test, y_train, y_test: Split data.
        '''
        logger.info("Splitting data using Stratified Split...")

        X, y = self.get_features_and_target(df, target_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test
    
# context class
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        '''
        Initialize the context with a specific strategy.

        Parameters:
            strategy (DataSplittingStrategy): The strategy to use.
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        '''
        Set a new strategy.

        Parameters:
            strategy (DataSplittingStrategy): The new strategy to use.
        '''
        self.strategy = strategy
    
    def split_data(self, df: pd.DataFrame, target_column: str):
        '''
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        The training and testing splits for features and target.
        '''
        logger.info(f"Splitting data using specified strategy: {self.strategy}")
        return self.strategy.split_data(df, target_column)