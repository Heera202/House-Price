from step.data_ingestion_step import data_ingestion_step
from step.data_spliting import data_spliting
from step.feature_engineer import feature_engineer
from step.handle_missing import handle_missing
from step.model_buildd import model_buildd
from step.model_eval import model_eval
from step.outlier_step import outlier_detection_step
import logging
from zenml import Model, pipeline, step
from step.debug_data_flow import debug_data_flow

from zenml import step

@pipeline(
    model = Model(
        name="prices_predictor"
    ),
    
)
def ml_pipeline():
    
    raw_data = data_ingestion_step(
        file_path="/Installed Softwares/House Price/Data/archive.zip"
    )
    # DEBUG after ingestion
    raw_data = debug_data_flow(raw_data, "AFTER INGESTION")
    
    filled_data = handle_missing(raw_data)
    
    engineer_data =feature_engineer(
        filled_data, strategy="log", features=["Gr Liv Area"]
    )
    
    clean_data = outlier_detection_step(engineer_data, column_name="SalePrice")
    
    X_train, X_test, y_train, y_test = data_spliting(clean_data, target_column="SalePrice")
    #X_train, X_test, y_train, y_test = debug_data_leakage(X_train, X_test, y_train, y_test)
        
    model = model_buildd(X_train =X_train, y_train=y_train)
    
    evaluation_metrics, mea = model_eval(trained_model=model, X_test=X_test, y_test=y_test)
    

    
    return model

if __name__ == "__main__":
    run = ml_pipeline()