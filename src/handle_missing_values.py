from abc import ABC, abstractmethod
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Base class
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Handling missing values in the data frame

        parameters:
        df (pd.DataFrame): Data Frame with missing values

        return:
        data frame after handling missing values
        '''
        pass

# concrete classes
class DropMissingValues(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        '''
        Initialize the strategy with specific parameters

        parameters:
        axis (int): 0 to drop rows with missing value, 1 to drop columns with missing value
        thresh (int): thresh hold for droping NA-values
        '''
        self.axis = axis
        self.thresh = thresh
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Drop rows or columns with missing values.

        Parameters:
            df (pd.DataFrame): The DataFrame with missing values.

        Returns:
            pd.DataFrame: The DataFrame after dropping missing values.
        '''
        logging.info(f"dropping na values with axix={self.axis} and thresh:{self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped")
        return df_cleaned

class FillMisssingValue(MissingValueHandlingStrategy):
    def __init__(self, method='mean', fill_value=None):
        '''
        Initializing missing value strategy with sepcific method

        parameters:
        method (str): which method to use - mean, median, mode
        fill_value (any): any value to fill in the data frame
        '''
        self.method = method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Impute missing value with mean, median, mode or any other value

        paramters:
        df (pd.DataFrame): Data frame with missing value

        return:
        Data frame with replaced missing vlaues
        '''
        logging.info(f"filling missing values using {self.method}")

        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include='number').columns

        if self.method == 'mean':
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        
        elif self.method == 'median':
           df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
               df[numeric_columns].median()
           )
        
        elif self.method == 'mode':
            for column in numeric_columns:
                df_cleaned[column].fillna(
                    df[column].mode().iloc(0),
                    inplace=True
                )
        
        elif self.method == 'constant':
            df_cleaned[numeric_columns].fillna(value=self.fill_value)
        
        else:
            logging.warning(f"Unknown method {self.method}. No Missing value handled")
        
        logging.info("Missing values handled")
        return df_cleaned


# context class
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        '''
        Initialize Missing value hanling strategy

        parameters:
        strategy (MissingValueHandlingStrategy)
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        '''
        set new strategy to hanling missing values

        parameters:
        strategy (MissingValueHandlingStrategy)
        '''
        self.strategy = strategy
    
    def handling_missing_value(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Handle missing values with specified strategies

        parameters:
        df (pd.DataFrame): The data frame that has missing values

        return:
        Pandas data frame that handled missing values
        '''
        return self.strategy.handle(df)