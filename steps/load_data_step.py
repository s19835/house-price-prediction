from src.load_data import DataProcessorFactory
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
    file_extension = file_path[file_path.rfind('.'):]
    raw_data = DataProcessorFactory.get_processor(file_extension)

    df = raw_data.load_data(file_path)
    return df
