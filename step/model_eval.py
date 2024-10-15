import logging
from src.model_evaluation import ModelEvaluation, RegressionModelEvaluation
from typing import Tuple
from sklearn.pipeline import Pipeline
from zenml import step
import pandas as pd

@step(enable_cache=False)
def model_eval(trained_model:Pipeline, X_test:pd.DataFrame, y_test:pd.DataFrame)->Tuple[dict,float]:
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    
    
    logging.info("Applying preprocessor to the test data")
    
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)
    
    evaluator = ModelEvaluation(strategy = RegressionModelEvaluation())
    evaluation_metrics = evaluator.evaluate(trained_model.named_steps["model"], X_test_processed, y_test)
    
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse