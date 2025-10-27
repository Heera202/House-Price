import os
from pipelines.training_pipeline import ml_pipeline
from step.dynamic_importer import dynamic_importer
from step.model_loader import model_loader
from step.prediction_service import prediction_service
from step.predictor import predictor

from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_deployer
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@pipeline(enable_cache=False)
def continuous_deployment_pipeline():
    trained_model= ml_pipeline()
    
    '''mlflow_deployer(workers=3, deploy_decision=True, model =trained_model)'''
      # Use the correct function name and fix parameter names
    mlflow_model_deployer_step(
        model=trained_model,
        workers=3,
        timeout=300,  # Add timeout parameter
    )


@pipeline(enable_cache=False)
def inference_pipeline():
    batch_data = dynamic_importer()
    
    
    model_deployment_service = prediction_service(
        pipeline_name = "continuous_deployment_pipeline",
        step_name ="mlflow_model_deployer_step",
    )
    
    predictor(service=model_deployment_service, input_data=batch_data)