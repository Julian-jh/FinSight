import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def add_sma(self, column: str = 'Close', window: int = 20, name: str = None):
        if name is None:
            name = f'SMA_{window}'
        self.df[name] = self.df[column].rolling(window=window).mean()
        logger.info(f"Added {name}")
        return self
    
    def add_ema(self, column: str = 'Close', span: int = 20, name: str = None):
        if name is None:
            name = f'EMA_{span}'
        self.df[name] = self.df[column].ewm(span=span, adjust=False).mean()
        logger.info(f"Added {name}")
        return self
    
    def add_rsi(self, column: str = 'Close', period: int = 14, name: str = 'RSI'):
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.df[name] = 100 - (100 / (1 + rs))
        logger.info(f"Added {name}")
        return self
    
    def add_bollinger_bands(self, column: str = 'Close', window: int = 20, num_std: float = 2):
        rolling_mean = self.df[column].rolling(window=window).mean()
        rolling_std = self.df[column].rolling(window=window).std()
        
        self.df['BB_Middle'] = rolling_mean
        self.df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        self.df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        logger.info("Added Bollinger Bands")
        return self
    
    def add_macd(self, column: str = 'Close', fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = self.df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df[column].ewm(span=slow, adjust=False).mean()
        
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        logger.info("Added MACD")
        return self
    
    def add_volatility(self, column: str = 'Close', window: int = 20, name: str = None):
        if name is None:
            name = f'Volatility_{window}'
        returns = self.df[column].pct_change()
        self.df[name] = returns.rolling(window=window).std()
        logger.info(f"Added {name}")
        return self
    
    def add_returns(self, column: str = 'Close', periods: int = 1, name: str = None):
        if name is None:
            name = f'Returns_{periods}'
        self.df[name] = self.df[column].pct_change(periods=periods)
        logger.info(f"Added {name}")
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_sma(window=50)
    indicators.add_ema(span=20)
    indicators.add_rsi()
    indicators.add_bollinger_bands()
    indicators.add_macd()
    indicators.add_volatility()
    indicators.add_returns()
    
    result = indicators.get_dataframe()
    logger.info(f"Features added. Shape: {result.shape}")
    logger.info(f"Columns: {result.columns.tolist()}")
