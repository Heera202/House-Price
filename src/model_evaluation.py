import logging
from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
import statsmodels.api as sm

import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model:Pipeline, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        pass
    
    
class RegressionModelEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, model:Pipeline, X_test:pd.DataFrame, y_test:pd.Series)->dict:
          # Make predictions
        
        y_pred = model.predict(X_test)
       


        '''logging.info(f"Predictions: {y_pred[:5]}")  # Log first 5 predictions'''

        logging.info("Calculating evaluation metrics.")
        

        # Log true values for reference
        '''logging.info(f"y_test: {y_test[:5]}")'''

        
        mea = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        

        metrics = {'Mean Absolute Error': mea, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    
    
class ModelEvaluation:
    def __init__(self, strategy:ModelEvaluationStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy:ModelEvaluationStrategy):
        logging.info("Swtiching model evaluation strategy")
        self._strategy = strategy
        
    def evaluate(self, model:RegressorMixin, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        logging.info("Evaluating the model using selected strategy")
        return self._strategy.evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":

    pass