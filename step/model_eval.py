'''import logging
from typing import Tuple
from sklearn.pipeline import Pipeline
from zenml import step
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@step(enable_cache=False)
def model_eval(trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, float]:
    """Evaluate the model using the full pipeline."""
    
    logging.info("Evaluating model on test data")
    
    # DEBUG: Check input data
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    logging.info(f"X_test columns: {X_test.columns.tolist()}")
    
    # CRITICAL FIX: Use the FULL pipeline for prediction
    # This ensures the same preprocessing as during training
    y_pred = trained_model.predict(X_test)
    
    # DEBUG: Check predictions vs actual values
    logging.info(f"y_pred shape: {y_pred.shape}")
    logging.info(f"First 5 predictions: {y_pred[:5]}")
    logging.info(f"First 5 actual values: {y_test.values[:5]}")
    
    # Check if predictions are identical to actual values
    if np.array_equal(y_pred, y_test.values):
        logging.error("🚨 CRITICAL: Predictions are IDENTICAL to actual values!")
        logging.error("This indicates serious data leakage!")
    else:
        logging.info("✅ Predictions are different from actual values")
    
    # Calculate metrics directly (bypass your evaluator for now)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    evaluation_metrics = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-Squared': r2
    }
    
    logging.info(f"Model Evaluation Metrics: {evaluation_metrics}")
    logging.info(f"MAE: {mae}")
    
    return evaluation_metrics, mae'''
    
    
import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluation import ModelEvaluation, RegressionModelEvaluation
from zenml import step


@step(enable_cache=False)
def model_eval(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply the preprocessing and model prediction
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluation(strategy=RegressionModelEvaluation())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse
