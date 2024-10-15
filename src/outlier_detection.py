import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df:pd.DataFrame)->pd.DataFrame:
        pass
    
    
class ZSCoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold
        
    def detect_outliers(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("Detecting Outliers using Z-score method")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-scores threshold: {self.threshold}.")
        return outliers
    
    
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self , df:pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outlier detected using the IQR mothod.")
        return outliers
    
    
class OutlierDetectorr:
    def __init__(self, strategy:OutlierDetectionStrategy):
        self._strategy = strategy
        
    def set_stratefy(self, strategy:OutlierDetectionStrategy):
        logging.info("Switching feature OutlierDetection strategy")
        self._strategy = strategy
        
    def detect_outliers(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("Applying feature OutlierDetection strategy.")
        return self._strategy.detect_outliers(df)
    
    def handle_outliers(self, df:pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers in the dataset")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df
        
        logging.info("Outlier handling completed.")
        return df_cleaned
    
    
    def visualize_outliers(self, df:pd.DataFrame, features:list):
        logging.info(f"Visualizing outliers for features:{features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x = df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed")
        
if __name__ =="__main__":
    pass
        
    
        
    

        
        
