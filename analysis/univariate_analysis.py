from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# base class (interface)
class UnivariateAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        '''
        perform univariate analysis in a single column of data

        parameters:
        df (pd.DataFrame): The data frame that feature exist
        feature (str): feature that you are going to analyze

        return:
        None: visualize the distribution of the feature
        '''
        pass

# concrete classes
class NumericalUnivariateAnalysis(UnivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature: str):
        '''
        perform univariate analysis on numerical data

        parameters:
        df (pd.DataFrame): The data frame that feature exist
        feature (str): feature that you are going to analyze

        return:
        None: display a histogram with kde
        '''
        logger.info("=== Numerical Univariate Analysis ===")
        
        sns.histplot(data=df[feature], kde=True, bins=30)
        
        plt.title(f"Distribution of {feature}")
        plt.xlabel('feature')
        plt.ylabel("frequency")
        plt.show()

class CategoricalUnivariateAnalysis(UnivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature: str):
        '''
        perform a categorical univariate analysis

        parameters:
        df (pd.DataFrame): Data frame that need to be analysis
        feature (str): categorical feature for analysis

        return:
        None: provide a visualization of the provided categorical variable
        '''
        sns.countplot(df, x=feature, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

# context class
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysis):
        '''
        initialize Univariate Analyze with specific strategy

        parameters:
        strategy (UnivariateAnalysis): strategy to use
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy: UnivariateAnalysis):
        '''
        set a new strategy to use

        parameters:
        strategy (UnivariateAnalysis): new strategy to set
        '''
        self.strategy = strategy
    
    def analyze(self, df: pd.DataFrame, feature: str):
        '''
        perform univariate analyze using current strategy

        parameters:
        df (pd.DataFrame): data frame that need to analyze
        feature (str): numerical feature to analyze

        return:
        None: Provide visualized plots for univariate data
        '''
        self.strategy.analyze(df, feature)
