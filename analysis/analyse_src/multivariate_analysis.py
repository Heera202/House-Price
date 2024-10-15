from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateAnalysisStrategy(ABC):

    def analyze(self, df:pd.DataFrame, target_column):
        self.generate_pairplot(df, target_column)
        self.generate_correlation_heatmap(df, target_column)
        
    
    @abstractmethod
    def generate_correlation_heatmap(self, df:pd.DataFrame, target_column):
        pass
    
    @abstractmethod
    def generate_pairplot(self, df:pd.DataFrame, target_column):
        pass
        
class SimpleMultivariateAnalysis(MultivariateAnalysisStrategy):
    def generate_correlation_heatmap(self, df:pd.DataFrame, target_column):
        
        df_numeric = df.select_dtypes(include=['number'])
        correlation_matrix = df_numeric.corr()
        corr_with_target = correlation_matrix[[target_column]].sort_values(by=target_column, ascending=False)
        
        plt.figure(figsize=(18, 15))
        sns.heatmap(corr_with_target, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df:pd.DataFrame, target_column):
        
        df_numeric = df.select_dtypes(include=['number'])
        plt.figure(figsize=(25, 20))
        sns.pairplot(df_numeric, hue=target_column)
        plt.suptitle("Pair plot of selected features", y=1.02)
        plt.show()
        

if __name__ == "__main__":
    pass
        
