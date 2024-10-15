import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStraṭegy,
    FillMissingValuesStrategy,
    MissingValueHandler
)

from zenml import step

@step
def handle_missing(df:pd.DataFrame, strategy:str='mean')->pd.DataFrame:
    
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStraṭegy(axis=0))
    elif strategy in ["mean", "mode", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value strategy:{strategy}")
    
    cleaned_df = handler.execute_strategy(df)
    return cleaned_df
            