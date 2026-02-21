import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_data(
    ticker: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    n_samples: int = 1000
):
    logger.info(f"Generating synthetic data for {ticker}")
    
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    
    data = pd.DataFrame({
        'Date': date_range,
        'Open': prices + np.random.randn(n_samples) * 0.5,
        'High': prices + np.abs(np.random.randn(n_samples) * 1),
        'Low': prices - np.abs(np.random.randn(n_samples) * 1),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples),
        'Adj Close': prices
    })
    
    logger.info(f"Generated {len(data)} records")
    return data


if __name__ == "__main__":
    data = generate_synthetic_data()
    
    output_path = Path("data/SPY_raw.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    
    logger.info(f"Saved to {output_path}")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    print("\nFirst few rows:")
    print(data.head())
