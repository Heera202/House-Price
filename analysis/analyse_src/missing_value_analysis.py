from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df:pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        
    @abstractmethod
    def identify_missing_values(self, df:pd.DataFrame):
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df:pd.DataFrame):
        pass
    
class SimpleMissingValueAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df:pd.DataFrame):
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
    def visualize_missing_values(self, df:pd.DataFrame):
        print("\nVisualize Missing Valuesssss...")
        plt.figure(figsize=(16, 11))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()
        
if __name__ == "__main__":
    df = pd.read_csv("../extracted_data/AmesHousing.csv")