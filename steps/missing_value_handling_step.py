import pandas as pd
from src.handle_missing_values import (
    DropMissingValues,
    FillMisssingValue,
    MissingValueHandler
)
from zenml import step

@step
def missing_value_handling_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    if strategy == 'drop':
        handler = MissingValueHandler(DropMissingValues(axis=0))
    
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMisssingValue(method=strategy))
    
    else:
        raise ValueError("Unsupported missing value handling strategy")
    
    cleaned_df = handler.handling_missing_value(df)
    return cleaned_df