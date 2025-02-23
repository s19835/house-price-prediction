from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# base class (interface)
class BivariateAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        '''
        analyze and compare bivariate data

        parameters:
        df (pd.DataFrame): Data frame that need to be analyzed
        feature1 (str): feature need to be comapred
        feature2 (str): feature need to be compared

        return:
        None: visualize the relationship between two features
        '''
        pass

# concrete class
class NumericalNumericalAnalysis(BivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        '''
        compare two numerical features

        parameters:
        df (pd.DataFrame): Data frame that need to be analyzed
        feature1 (str): numerical feature need to be comapred
        feature2 (str): numerical feature need to be compared
        
        return:
        None: Provide a scatterplot with specified numerical features
        '''
        sns.scatterplot(data=df, x=feature1, y=feature2, color='blue')
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
class NumericalCategoricalAnalysis(BivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        '''
        compare numerical with categorical data

        parameters:
        df (pd.DataFrame): data frame to analyze
        feature1 (str): numerical feature need to be comapred
        feature2 (str): categorical feature need to be compared

        return:
        None: display a boxplot showing the relationship between numerical and categorical feature
        '''
        sns.boxplot(data=df, x=feature2, y=feature1)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()
    
class CategoricalCategoricalAnalysis(BivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        '''
        analyze categorical feature with another categorical feature

        parameters:
        df (pd.DataFrame): data frame to analyze
        feature1 (str): categorical feature need to be comapred
        feature2 (str): categorical feature need to be compared
        
        
        '''
        pass

# context class
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysis):
        '''
        initialize the bivariate with specific anlaysis strategy

        parameters:
        strategy (BivariateAnalysis): The analysis strategy to use
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: BivariateAnalysis):
        '''
        set new strategy

        parameters:
        strategy (BivariateAnalysis): The new analyze strategy to use
        '''
        self.strategy = strategy
    
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        '''
        perform bivariate anlysis using current strategy

        parameters:
        df (pd.DataFrame): data frame to analyze
        feature1 (str): feature need to be comapred
        feature2 (str): feature need to be compared
        '''
        self.strategy.analyze(df, feature1, feature2)