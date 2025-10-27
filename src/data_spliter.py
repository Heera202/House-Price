import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df:pd.DataFrame, target_column:str):
        pass
    
    
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size = 0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        
    def split_data(self, df:pd.DataFrame, target_column:str):
        logging.info("Performing simple train-test split")
        
        '''X, y = make_regression(n_samples=100, n_features=10, noise=0.1)'''
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        
        
        '''k = 5
        kf = KFold(n_splits = k, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
            
        return X_train , X_test , y_train, y_test ''' 
        
        
        
        '''X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state = self.random_state)
        
        logging.info("Train-test split completed")
        
        return X_train , X_test , y_train, y_test'''
             # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        logging.info(f"Train set: {X_train.shape}")
        logging.info(f"Test set: {X_test.shape}")
        logging.info("Train-test split completed")
        
        return X_train, X_test, y_train, y_test
        
        
    
    
class DataSplitter:
    def __init__(self, strategy:DataSplittingStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy:DataSplittingStrategy):
        self._strategy = strategy
        
    def split(self, df:pd.DataFrame, target_column:str):
        logging.info("Spliting data unsing the selected strategy")
        return self._strategy.split_data(df, target_column)
    
    
if __name__ == "__main__":
    pass
        
        