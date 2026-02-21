import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.fetch_data import DataFetcher
from src.validation.validate import DataValidator
from src.features.technical_indicators import TechnicalIndicators
from src.features.lagged_features import LaggedFeatures
from src.models.train import ModelTrainer
from src.utils.logger import setup_logger
from src.utils.helpers import ensure_dir

logger = setup_logger(__name__)


class FinSightPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data = None
        self.model_trainer = None
        
    def run(self, ticker: str = "SPY", start_date: str = "2020-01-01", end_date: str = None):
        logger.info("="*50)
        logger.info("Starting FinSight Pipeline")
        logger.info("="*50)
        
        # Step 1: Data Ingestion
        logger.info("\n[1/5] Data Ingestion")
        fetcher = DataFetcher(data_dir=self.config.get('data_dir', 'data'))
        
        # Try to load existing data first, then fetch if not available
        try:
            self.data = fetcher.load_raw_data(f"{ticker}_raw.csv")
            logger.info(f"Loaded existing data from {ticker}_raw.csv")
        except FileNotFoundError:
            self.data = fetcher.fetch_stock_data(ticker, start_date, end_date)
            if not self.data.empty:
                fetcher.save_raw_data(self.data, f"{ticker}_raw.csv")
            else:
                logger.error("No data available. Please generate test data first.")
                raise ValueError("No data available")
        
        # Step 2: Data Validation
        logger.info("\n[2/5] Data Validation")
        validator = DataValidator()
        validation_report = validator.validate_data(self.data)
        
        # Step 3: Feature Engineering
        logger.info("\n[3/5] Feature Engineering")
        indicators = TechnicalIndicators(self.data)
        indicators.add_sma(window=20).add_sma(window=50)
        indicators.add_rsi().add_returns()
        self.data = indicators.get_dataframe()
        
        # Step 4: Model Training
        logger.info("\n[4/5] Model Training")
        self.model_trainer = ModelTrainer(config=self.config.get('model', {}))
        df_with_target = self.model_trainer.create_target(self.data)
        X, y = self.model_trainer.prepare_data(df_with_target)
        X_train, X_test, y_train, y_test = self.model_trainer.train_test_split_temporal(X, y)
        
        model = self.model_trainer.train(X_train, y_train, model_type='random_forest')
        
        # Step 5: Save Model
        logger.info("\n[5/5] Saving Model")
        ensure_dir('models')
        self.model_trainer.save('models/model.pkl', 'models/scaler.pkl')
        
        logger.info("\n" + "="*50)
        logger.info("Pipeline Completed Successfully")
        logger.info("="*50)
        
        return self.data, model


if __name__ == "__main__":
    pipeline = FinSightPipeline()
    data, model = pipeline.run(ticker="SPY", start_date="2020-01-01")
