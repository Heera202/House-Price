import logging
from zenml import step
import pandas as pd

@step(enable_cache=False)
def debug_data_flow(df: pd.DataFrame, step_name: str, target_column: str = "SalePrice"):
    """Debug data at each step of the pipeline."""
    logging.info(f"🔍 === DEBUG {step_name} ===")
    logging.info(f"DataFrame shape: {df.shape}")
    
    if target_column in df.columns:
        logging.info(f"Target '{target_column}' dtype: {df[target_column].dtype}")
        logging.info(f"Target unique values: {df[target_column].unique()[:5]}")
        logging.info(f"Target sample values: {df[target_column].head(5).tolist()}")
    else:
        logging.warning(f"Target column '{target_column}' not found!")
    
    logging.info(f"All columns: {df.columns.tolist()}")
    return df