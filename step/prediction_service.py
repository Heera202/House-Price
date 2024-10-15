from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowModelDeploymentService

@step(enable_cache=False)
def prediction_service(pipeline_name:str, step_name:str)->MLFlowModelDeploymentService:
    
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No Mlflow prediction serivce deployed by the"
            f"{step_name} step in the {pipeline_name}"
            f"pipeline is currently "
            f"running."
        )
    return existing_services[0]