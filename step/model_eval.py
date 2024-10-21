import logging
from src.model_evaluation import ModelEvaluation, RegressionModelEvaluation
from typing import Tuple
from sklearn.pipeline import Pipeline
from zenml import step
import pandas as pd
import numpy as np

@step(enable_cache=False)
def model_eval(trained_model:Pipeline, X_test:pd.DataFrame, y_test:pd.Series)->Tuple[dict, float]:
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    
    
    logging.info("Applying preprocessor to the test data")
    
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)
    
    if np.any(np.isnan(X_test_processed)):
        logging.error("Transformed test data contains NaN values.")
        raise ValueError("Transformed test data contains NaN values.")

    
    evaluator = ModelEvaluation(strategy=RegressionModelEvaluation())
    try:
        evaluation_metrics = evaluator.evaluate(trained_model.named_steps["model"], X_test_processed, y_test)
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise
    
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mea = evaluation_metrics.get("Mean Absolute Error", None)
    
    if mea is None:
        logging.error("Mean Absolute Error was not computed correctly.")
        raise ValueError("Mean Absolute Error is None. Please check the evaluation metrics.")
    
    logging.info(f"MAE: {mea}")
    
    return evaluation_metrics, mea