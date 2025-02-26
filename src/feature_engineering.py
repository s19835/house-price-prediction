import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# define a base class for feature engineering strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Transform the DataFrame using the feature engineering strategy.

        Parameters:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        '''
        pass

# create base class with strategies
# log transformation mostly to target variable
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        '''
        initialize log transformation with specific features

        parameters:
        feature (str): list of feature that need to be transformed
        '''
        self.features = features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        apply log transformation to the specified features

        parameters:
        df (pd.DataFrame): pandas data frame that need feature engineering

        return: pandas data frame with transformed features
        '''
        logger.info(f"Apply log transformation for the features: {self.features}")
        
        transformed_data = df.copy()
        for feature in self.features:
            transformed_data[feature] = np.log1p(
                df[feature]
            )
        
        logger.info("log transformation completed")
        return transformed_data

# standard scaling
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        '''
        initialize standard scaling strategy with specific features

        parameters:
        features (list): list of features to apply scaling
        '''
        self.features = features
        self.scaler = StandardScaler()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        apply standard scaling to a specific features

        parameters:
        df (pd.DataFrame): data frame that we can apply standard scaling

        return: 
        (pd.DataFrame): pandas data frame with scaled values
        '''
        transformed_data = df.copy()

        logger.info("Applying Standard Scaling...")
        transformed_data[self.features] = self.scaler.fit_transform(df[self.features])
        return transformed_data

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list, feature_range=(0, 1)):
        '''
        initialize the min max scaling strategy

        parameters: 
        features (list): list of features for scaling
        feature_range (tuple): the target range for scaling, default (0, 1)
        '''
        self.features = features
        self.scaler = MinMaxScaler(feature_range)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        apply min max scaling to the provided data frame

        parameters:
        df (pd.DataFrame): Data frame that need to be scaled

        return:
        (pd.DataFrame): pandas data frame with scaled values
        '''
        logger.info("Applying Min Max Scaling...")
        logger.info(f"Scale {self.features} in a range of {self.scaler.feature_range}")

        scaled_data = df.copy()
        scaled_data[self.features] = self.scaler.fit_transform(df[self.features])
        return scaled_data

# one hot encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        '''
        initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        '''
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        '''
        logger.info("Appling One Hot Encoder to provided categorical features")

        transformed_data = df.copy()

        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )

        transformed_data = transformed_data.drop(columns=self.features).reset_index(drop=True)
        transformed_data = pd.concat([transformed_data, encoded_df], axis=1)
        return transformed_data
    
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        '''
        initialize feature engineering strategy

        parameters:
        strategy (FeatureEngineeringStrategy): strategy to use in feature engineering
        '''
        self.strategy = strategy
    
    def set_strategy(self, strategy:FeatureEngineeringStrategy):
        '''
        set specific strategy to use

        parameters:
        strategy (FeatureEngineeringStrategy): new strategy to use in feature engineering
        '''
        self.strategy = strategy
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        '''
        logger.info("Applying Feature Engineering...")
        return self.strategy.transform(df)