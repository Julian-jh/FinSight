import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LaggedFeatures:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def add_lags(self, columns: List[str], lags: List[int]):
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column {col} not found")
                continue
            
            for lag in lags:
                lag_name = f"{col}_lag_{lag}"
                self.df[lag_name] = self.df[col].shift(lag)
                logger.info(f"Added {lag_name}")
        
        return self
    
    def add_rolling_features(self, column: str, windows: List[int], agg_funcs: List[str] = ['mean', 'std']):
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return self
        
        for window in windows:
            for func in agg_funcs:
                feature_name = f"{column}_rolling_{func}_{window}"
                if func == 'mean':
                    self.df[feature_name] = self.df[column].rolling(window=window).mean()
                elif func == 'std':
                    self.df[feature_name] = self.df[column].rolling(window=window).std()
                elif func == 'min':
                    self.df[feature_name] = self.df[column].rolling(window=window).min()
                elif func == 'max':
                    self.df[feature_name] = self.df[column].rolling(window=window).max()
                
                logger.info(f"Added {feature_name}")
        
        return self
    
    def add_diff_features(self, columns: List[str], periods: List[int] = [1]):
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column {col} not found")
                continue
            
            for period in periods:
                diff_name = f"{col}_diff_{period}"
                self.df[diff_name] = self.df[col].diff(periods=period)
                logger.info(f"Added {diff_name}")
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    from src.features.technical_indicators import TechnicalIndicators
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_rsi()
    data_with_indicators = indicators.get_dataframe()
    
    lagged = LaggedFeatures(data_with_indicators)
    lagged.add_lags(['Close', 'Volume'], lags=[1, 5, 10])
    lagged.add_rolling_features('Close', windows=[5, 10, 20], agg_funcs=['mean', 'std'])
    lagged.add_diff_features(['Close'], periods=[1, 5])
    
    result = lagged.get_dataframe()
    logger.info(f"Final shape: {result.shape}")
    logger.info(f"Total columns: {len(result.columns)}")
