from abc import ABC, abstractmethod
import pandas as pd


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df:pd.DataFrame):
    
        pass
    

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        print("\nData types and Non Null counts:")
        print(df.info())
        
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        print("Summary Statistics(NUmerical Features)")
        print(df.describe())
        print("Summary Statistics(Categorical Features)")
        print(df.describe(include =['O']))
        
class DataInspector:
    def __init__(self, strategy:DataInspectionStrategy):
        self._strategy =strategy
        
    def set_strategy(self, strategy:DataInspectionStrategy):
        self._strategy = strategy
        
    def execute_inspection(self, df:pd.DataFrame):
        self._strategy.inspect(df)
        
if __name__ == "__main__":
    df = pd.read_csv("extracted_data\AmesHousing.csv")
    
    inspector = DataInspector(DataInspectionStrategy())
    inspector.execute_inspection(df)
    
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)
    pass