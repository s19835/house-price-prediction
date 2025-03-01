import click
from training.training_pipline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@click.command()
def main():
    """
    Run the ML flow Pipline and kick strat the MLflow UI dashborad for experiment tracking
    """

    # run the pipline
    run = ml_pipeline()

    logger.info(f"\nmlflow UI --backend-stroe-ui '{get_tracking_uri()}'\n")

if __name__ == "__main__":
    main()