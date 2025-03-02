import os
from zenml import pipeline

from training.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')

@pipeline
def continuous_deployment_pipeline():
    '''Run trainning and deploy mlflow model'''

    # run trainning pipeline
    trained_model = ml_pipeline()

    # deploy the model using mlflow
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)

@pipeline(enable_cache=False)
def inference_pipeline():
    '''Run batch inference with data from api'''

    # load the batch data for inference
    batch_data = dynamic_importer()

    # load deployment model service
    model_development_service = prediction_service_loader(
        pipeline_name='continuous_deployment_pipeline',
        step_name = 'mlflow_model_deployer_step'
    )

    # run prediction on batch size
    predictor(service=model_development_service, input_data=batch_data)
    