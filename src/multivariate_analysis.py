from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# base class (template interface)
class MultivariateAnalysis(ABC):
    def analyze(self, df: pd.DataFrame):
        '''
        performing comprehensive multivariate anlysis on data set

        parameters:
        df (pd.DataFrame): pandas Data frame that need to be analyzed

        return: 
        None: produce a correlation heatmap and pairplot
        '''
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        '''
        generate correlation heatmap for the data frame

        parameters:
        df (pd.DataFrame): data frame need to be analyzed

        return:
        None: generate a heatmap
        '''
        pass
    
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        '''
        generate a pairplot from the data frame

        parameters:
        df (pd.DataFrame): data frame need to be analyzed

        return:
        None: generate a pairplot
        '''
        pass

# concreate classes
class BasicMultivariateAnalysis(MultivariateAnalysis):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        '''
        generate correlation heatmap for the data frame

        parameters:
        df (pd.DataFrame): data frame need to be analyzed

        return:
        None: generate a heatmap
        '''
        sns.heatmap(data=df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
    
    def generate_pairplot(self, df: pd.DataFrame):
        '''
        generate a pairplot from the data frame

        parameters:
        df (pd.DataFrame): data frame need to be analyzed

        return:
        None: generate a pairplot
        '''
        sns.pairplot(df)
        plt.suptitle("Pairplot with selected features", y=1.02)
        plt.show()