from zenml import Model, pipeline
from steps.load_data_step import data_load_step

@pipeline(
    model=Model(
        name="price-prediction"
    )
)

def ml_pipline():
    '''Define an end to end ML pipline'''

    # load data step
    data = data_load_step(
        "./data/raw/test.csv"
    )