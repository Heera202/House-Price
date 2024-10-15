import pandas as pd
from typing import Tuple

from src.data_spliter import SimpleTrainTestSplitStrategy, DataSplitter

from zenml import step

@step
def data_spliting(df:pd.DataFrame, target_column:str)->Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_tests = splitter.execute_strategy(df, target_column)
    return  X_train, X_test, y_train, y_tests