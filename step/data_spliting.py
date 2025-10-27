import pandas as pd
from typing import Tuple

from src.data_spliter import SimpleTrainTestSplitStrategy, DataSplitter

from zenml import step
import logging

@step
def data_spliting(df:pd.DataFrame, target_column:str)->Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    
  
    return  X_train, X_test, y_train, y_test

