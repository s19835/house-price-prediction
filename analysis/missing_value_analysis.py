from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# base class
class MissingValueAnalysis(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        '''
        Template method to perform complete missing value analysis

        parameters:
        df (pd.DataFrame) : data frame has missing values
        '''
        self.identify_missing_value(df)
        self.visualize_missing_value(df)

    @abstractmethod
    def identify_missing_value(self, df: pd.DataFrame):
        '''
        identify the missing values in the data frame

        parameters:
        df (pd.DataFrame): data frame has missing values

        return:
        None: print out the count of missing value
        '''
        pass

    @abstractmethod
    def visualize_missing_value(self, df: pd.DataFrame):
        '''
        visualize missing value in the data frame

        parameters:
        df (pd.DataFrame): data frame that has missing values

        return:
        None: provide visualization graphs like graphs, histogram, charts, heatmap etc
        '''
        pass

class SimpleMissingValueAnalysis(MissingValueAnalysis):
    def identify_missing_value(self, df: pd.DataFrame):
        '''
        identify missing values and count them
        '''
        logger.info("\nMissing value count by column")
        missing_values = df.isnull().sum()
        logger.info(missing_values[missing_values > 0])
    
    def visualize_missing_value(self, df: pd.DataFrame):
        '''
        visualize missing value using heatmap
        '''
        logger.info("\nVisualize missing value...")
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing value Heatmap")
        plt.show()
