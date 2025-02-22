from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Base class
class DataInspection(ABC):
    @abstractmethod
    def inspect_data(self, df: pd.DataFrame):
        '''
        perform a data inspection on basic level

        parameters:
        df (pd.DataFrame) : data frame need to be inspected

        returns: None
        '''
        pass

# Concrete class (strategies)
class DataTypeInspection(DataInspection):
    def inspect_data(self, df: pd.DataFrame):
        '''
        inspect data types in the dataframe

        parameters:
        df (pd.DataFrame) : data frame need to be inspected

        returns: Prints data types in the data frame
        '''

        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas data frame")
            return

        logger.info("=== Data Types in Data Frame ===")
        logger.info("\n" + str(df.info()))

class DataSummaryStatistics(DataInspection):
    def inspect_data(self, df: pd.DataFrame):
        '''
        provide summary statistics of the data frame

        parameters:
        df (pd.DataFrame) : data frame need to be summarized

        returns:
        summary statistic of data frame
        '''
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas data frame")
            return
        
        logger.info("=== Summary statistics ===")
        
        logger.info("\nNumerical Features: ")
        logger.info(df.describe())
        
        logger.info("\nCategorical Features: ")
        logger.info(df.describe(include=[object]))

# context class (to use diff statergies)
class DataInspector:
    def __init__(self, strategy: DataInspection):
        '''
        Initialize the data inspector with specific strategy

        parameters: 
        strategy (DataInspection): The inspection strategy to use
        '''
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataInspection):
        '''
        set new data inspection strategy to use

        parameters:
        strategy (DataInspection): The new inspection strategy to use
        '''
        self._strategy = strategy
    
    def inspect(self, df: pd.DataFrame) -> None:
        '''
        perform data inspection using the strategy

        parameters:
        df (pd.DataFrame): dataframe to be inspected

        return:
        print statement of stratgy out put (Summarized Data)
        '''
        self._strategy.inspect_data(df)