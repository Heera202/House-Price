import logging

import pandas as pd
from src.outlier_detection import(
    ZSCoreOutlierDetection,
    IQROutlierDetection,
    OutlierDetectorr  
)

from zenml import step

@step
def outlier_detection_step(df:pd.DataFrame, column_name:str):
    logging.info(f"Starting outlier detection step with DataFrame of shape:{df.shape}")
    
    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be Non Null pandas DataFrame")
    
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead")
        raise ValueError("Input df must be a pandas DataFrame")
    
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in DataFrame")
        raise ValueError(f"Column '{column_name}' does not exist in DataFrame")
    df_numeric = df.select_dtypes(include = [int, float])
    
    outlier_detector = OutlierDetectorr(strategy=ZSCoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(outliers, method="remove")
    
    return df_cleaned