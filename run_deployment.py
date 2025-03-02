import click

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from deployment.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline
)

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@click.command()
@click.option("--stop-service", is_flag=True, default=False, help='stop the prediction service when done')
def run_deployment(stop_service: bool):
    '''
    Run the deployment process

    parameters:
    stop_service (bool): flag to stop the running prediction
    '''
    model_name = 'price_prediction'

    if stop_service:
        # get mlflow model deployer stack
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # fetch existing serivices
        existing_services = model_deployer.find_model_server(
            pipeline_name='continuous_deployment_pipeline',
            pipeline_step_name='mlflow_model_deployer_step',
            model_name=model_name,
            model_version='0.1.0',
            running=True
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return
    
    # run the cts deployment pipline
    continuous_deployment_pipeline()

    # get active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # run inference pipline
    inference_pipeline()

    logger.info(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
        )
    
    # fetch existing serivices
    service = existing_services = model_deployer.find_model_server(
        pipeline_name='continuous_deployment_pipeline',
        pipeline_step_name='mlflow_model_deployer_step',
    )

    if service[0]:
        logger.info(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )

if __name__ == '__main__':
    run_deployment()