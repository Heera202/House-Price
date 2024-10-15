import logging
from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model:RegressorMixin, X_test:pd.DataFrame, y_test:pd.DataFrame):
        pass
    
    
class RegressionModelEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, model:RegressorMixin, X_test:pd.DataFrame, y_test:pd.DataFrame):
        logging.info("Predicting using the trained model")
        y_pred = model.predict(X_test)
        
        logging.info("Calculating evaluation metrics.")
        mae = mean_absolute_error(y_test, y_pred)
        r2 =r2_score(y_test, y_pred)
        
        metrics = {'Mean absolute Error':mae, "R-Squared": r2}
        
        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    
    
class ModelEvaluation:
    def __init___(self, strategy:ModelEvaluationStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy:ModelEvaluationStrategy):
        logging.info("Swtiching model evaluation strategy")
        self._strategy = strategy
        
    def evaluate(self, model, X_test, y_test):
        logging.info("Evaluating the model using selected strategy")
        return self._strategy.evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":
    pass