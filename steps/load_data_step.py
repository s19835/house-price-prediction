from src.load_data import load_file
from zenml import step

import pandas as pd

@step
def data_load_step(file_path: str) -> pd.DataFrame:
    '''
    load data from file as pandas dataframe

    parameters:
    file_path (str): path to the file

    return:
    pandas data frame with data from files
    '''
    df = load_file(file_path)
    return df
