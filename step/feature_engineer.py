import logging
from src.feature_engineering import LogTransformation,MinMaxScaling,StandardScaling, OneHotEncoding, FeatureEngineer
import pandas as pd

from zenml import step


@step
def feature_engineer(df:pd.DataFrame, strategy:str="log",features:list=None) -> pd.DataFrame:
    
    if features is None:
        features = []
    
    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
        
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
        
    elif strategy == "minmax_scaling":
        engineer =  FeatureEngineer(MinMaxScaling(features))
        
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy:{strategy}")
    
    transformed_df = engineer.apply_feature_engineering(df)
    
    return transformed_df
        
        
         