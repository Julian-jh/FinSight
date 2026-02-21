import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

logger = setup_logger(__name__)


class Backtester:
    def __init__(self, initial_window: int = 252, step_size: int = 21):
        self.initial_window = initial_window
        self.step_size = step_size
        self.results = []
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        config: Dict = None
    ) -> List[Dict]:
        n_samples = len(X)
        results = []
        
        logger.info(f"Starting walk-forward validation with initial window={self.initial_window}, step={self.step_size}")
        
        current_start = 0
        fold = 0
        
        while current_start + self.initial_window < n_samples:
            train_end = current_start + self.initial_window
            test_end = min(train_end + self.step_size, n_samples)
            
            if train_end >= n_samples or test_end > n_samples:
                break
            
            X_train = X.iloc[current_start:train_end]
            y_train = y.iloc[current_start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            logger.info(f"\nFold {fold + 1}: Train [{current_start}:{train_end}], Test [{train_end}:{test_end}]")
            
            trainer = ModelTrainer(config=config)
            trainer.train(X_train, y_train, model_type=model_type)
            
            X_test_scaled = trainer.scaler.transform(X_test)
            y_pred = trainer.model.predict(X_test_scaled)
            y_pred_proba = trainer.model.predict_proba(X_test_scaled)[:, 1]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            fold_results = {
                'fold': fold + 1,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            results.append(fold_results)
            logger.info(f"Fold {fold + 1} Results: Accuracy={fold_results['accuracy']:.4f}, F1={fold_results['f1']:.4f}")
            
            current_start += self.step_size
            fold += 1
        
        self.results = results
        self._log_summary()
        
        return results
    
    def _log_summary(self):
        if not self.results:
            return
        
        df_results = pd.DataFrame(self.results)
        
        logger.info("\n" + "="*50)
        logger.info("BACKTESTING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Folds: {len(self.results)}")
        logger.info(f"\nMean Metrics:")
        logger.info(f"  Accuracy: {df_results['accuracy'].mean():.4f} (+/- {df_results['accuracy'].std():.4f})")
        logger.info(f"  Precision: {df_results['precision'].mean():.4f} (+/- {df_results['precision'].std():.4f})")
        logger.info(f"  Recall: {df_results['recall'].mean():.4f} (+/- {df_results['recall'].std():.4f})")
        logger.info(f"  F1: {df_results['f1'].mean():.4f} (+/- {df_results['f1'].std():.4f})")
        logger.info("="*50)
    
    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    from src.features.technical_indicators import TechnicalIndicators
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_rsi().add_returns()
    df_with_features = indicators.get_dataframe()
    
    trainer = ModelTrainer()
    df_with_target = trainer.create_target(df_with_features)
    X, y = trainer.prepare_data(df_with_target)
    
    backtester = Backtester(initial_window=252, step_size=21)
    results = backtester.walk_forward_validation(X, y, model_type='random_forest')
    
    results_df = backtester.get_results()
    logger.info(f"\nResults saved. Shape: {results_df.shape}")
