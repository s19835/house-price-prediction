from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    StandardScaling,
    MinMaxScaling,
    OneHotEncoding,
)
import pandas as pd
from zenml import step

@step
def feature_engineering_step(
    df: pd.DataFrame, 
    strategy: str = 'log', 
    features: list = None) -> pd.DataFrame:
    '''Perform feature engineering using specified strategies'''

    if features is None:
        features = []
    
    if strategy == 'log':
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == 'standard-scaling':
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")
    
    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df