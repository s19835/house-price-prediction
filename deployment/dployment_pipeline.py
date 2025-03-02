import os
from zenml import pipeline

from training.training_pipeline import ml_pipeline

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')

@pipeline
def continuous_deployment_pipeline():
    '''Run trainning and deploy mlflow model'''

    # run trainning pipeline
    trained_model = ml_pipeline()

    # deploy the model using mlflow
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)

