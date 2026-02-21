import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    def __init__(self):
        self.validation_report = {}
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        result = {
            'total_missing': missing.to_dict(),
            'missing_percentage': missing_pct.to_dict()
        }
        
        if missing.sum() > 0:
            logger.warning(f"Missing values detected: {missing[missing > 0].to_dict()}")
        else:
            logger.info("No missing values found")
        
        return result
    
    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> Dict:
        duplicates = df.duplicated(subset=subset).sum()
        
        result = {
            'duplicate_count': int(duplicates),
            'duplicate_percentage': float((duplicates / len(df)) * 100)
        }
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return result
    
    def check_data_types(self, df: pd.DataFrame, expected_types: Dict = None) -> Dict:
        current_types = df.dtypes.to_dict()
        
        if expected_types:
            mismatches = {}
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    if str(df[col].dtype) != expected_type:
                        mismatches[col] = {
                            'expected': expected_type,
                            'actual': str(df[col].dtype)
                        }
            
            if mismatches:
                logger.warning(f"Data type mismatches: {mismatches}")
            return {'mismatches': mismatches, 'current_types': current_types}
        
        logger.info(f"Data types: {current_types}")
        return {'current_types': current_types}
    
    def check_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(df)) * 100),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                }
        
        logger.info(f"Outlier detection completed for {len(columns)} columns")
        return outliers
    
    def validate_data(self, df: pd.DataFrame, expected_types: Dict = None) -> Dict:
        logger.info("Starting data validation")
        
        report = {
            'shape': df.shape,
            'missing_values': self.check_missing_values(df),
            'duplicates': self.check_duplicates(df),
            'data_types': self.check_data_types(df, expected_types),
            'outliers': self.check_outliers(df)
        }
        
        self.validation_report = report
        logger.info("Data validation completed")
        
        return report
    
    def get_report(self) -> Dict:
        return self.validation_report


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    validator = DataValidator()
    report = validator.validate_data(data)
    
    logger.info(f"Validation report: {report}")
