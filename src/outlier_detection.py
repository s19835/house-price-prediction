# import necessary libraries
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# interface for the strategy - base class
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        detect outliers using several strategies

        parameters:
        df (pd.DataFrame): Data Frame that has outliers
        '''
        pass

# concreate classes with strategies

class ZScoreMethod(OutlierDetectionStrategy):
    def __init__(self, threshold = 3):
        '''
        Initialize z-score detection method with threshold value

        parameters:
        threshold (int): threshold for Z-Score
        '''
        self.threshold = threshold
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Detect outliers using the Z-score method.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            pd.DataFrame: A boolean indicating outliers (True = outlier).
        '''
        logger.info("Detecting Outliers Using Z-Score method...")

        z_score = np.abs((df - df.mean()) / df.std())
        outliers = z_score > self.threshold
        return outliers

class IQRMethod(OutlierDetectionStrategy):
    def __init__(self, multiplier: float = 1.5):
        '''
        Initialize the strategy with an IQR multiplier.

        Parameters:
            multiplier (float): The IQR multiplier for detecting outliers (default: 1.5).
        '''
        self.multiplier = multiplier
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Detect outliers using the IQR method.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            pd.DataFrame: A boolean indicating outliers (True = outlier).
        '''
        logger.info("Detect outliers using IQR Method...")

        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)

        IQR = Q3 - Q1
        outliers = (df < (Q1 - self.multiplier * IQR)) | (df > (Q3 - self.multiplier * IQR))
        return outliers
    
# context class
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        '''
        initialize outlier detection strategy

        parameters:
        strategy (OutlierDetectionStrategy): Strategy to use to detect outliers
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: OutlierDetectionStrategy):
        '''
        set new outlier detections strategy to use

        parameters:
        strategy (OutlierDetectionStrategy): new strategy to use
        '''
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        using specified strategy and detect outliers

        prameters:
        df (pd.DataFrame): Data Frame that need to be tested

        return:
        pd.DataFrame: pandas data frame with boolean values if outlier exist or not
        '''
        return self.strategy.detect_outliers(df)
    
    def handle_outliers(self, df: pd.DataFrame, method='remove', **kwargs) -> pd.DataFrame:
        '''
        by using methods like remove, cap handle outliers

        parameters:
        df (pd.DataFrame): Data Frame that has outliers
        method (str): either remove or cap
        **kwargs (any): any other inputs

        return:
        pd.DataFrame: pandas data frame without outliers
        '''
        outliers = self.detect_outliers(df)

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
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        '''
        visualize outliers for every feature

        parameters:
        df (pd.DataFrame): pandas data frame that has outliers
        features (list): list of feature that has outliers to visualize

        return:
        None: provide a box plot for every feature that show outliers
        '''
        logger.info(f"Visualizing outliers in features: {features}")
        
        for feature in features:
            sns.boxplot(x = df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
