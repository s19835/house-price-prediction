from abc import ABC, abstractmethod
import pandas as pd

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
        print("Data Types in Data Frame: \n")
        print(df.info())

class DataSummaryStatistics(DataInspection):
    def inspect_data(self, df: pd.DataFrame):
        '''
        provide summary statistics of the data frame

        parameters:
        df (pd.DataFrame) : data frame need to be summarized

        returns:
        summary statistic of data frame
        '''
        print("Summary statistics (Numerical Features): \n")
        print(df.describe())
        print("Summary statistics (Categorical Features): \n")
        print(df.describe(include=[object]))

