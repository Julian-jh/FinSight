# FinSight

Financial Machine Learning Pipeline for Market Regime Classification

## Overview

FinSight is a production-ready ML pipeline for predicting market regimes using technical indicators and time-series features. Built with proper separation of concerns, time-aware validation, and API-ready inference.

## Features

- 📊 **Data Ingestion**: Automated financial data download via yfinance
- ✅ **Data Validation**: Quality checks, missing value detection, outlier analysis
- 🔧 **Feature Engineering**: Technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands)
- 🤖 **Model Training**: Multiple models with time-series cross-validation
- 📈 **Backtesting**: Walk-forward validation to prevent data leakage
- 🚀 **REST API**: FastAPI inference endpoint for predictions

## Project Structure

```
finsight/
├── data/                    # Raw and processed data
├── configs/                 # Configuration files
│   └── config.yaml
├── notebooks/               # Exploratory analysis
├── src/
│   ├── ingestion/           # Data fetching
│   ├── validation/          # Data quality checks
│   ├── features/            # Feature engineering
│   ├── models/              # Training and evaluation
│   ├── backtesting/         # Walk-forward validation
│   ├── pipeline/            # End-to-end orchestration
│   ├── api/                 # FastAPI inference server
│   └── utils/               # Logging and helpers
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python src/pipeline/pipeline.py
```

### 3. Start the API

```bash
python src/api/main.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage Examples

### Fetch Data

```python
from src.ingestion.fetch_data import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_stock_data("SPY", "2020-01-01", "2024-12-31")
fetcher.save_raw_data(data, "SPY_raw.csv")
```

### Engineer Features

```python
from src.features.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators(data)
indicators.add_sma(window=20).add_rsi().add_macd()
df_features = indicators.get_dataframe()
```

### Train Model

```python
from src.models.train import ModelTrainer

trainer = ModelTrainer()
df_with_target = trainer.create_target(df_features)
X, y = trainer.prepare_data(df_with_target)
X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y)

model = trainer.train(X_train, y_train, model_type='random_forest')
trainer.save('models/model.pkl', 'models/scaler.pkl')
```

### Backtest

```python
from src.backtesting.backtest import Backtester

backtester = Backtester(initial_window=252, step_size=21)
results = backtester.walk_forward_validation(X, y)
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Get regime prediction

## Configuration

Edit `configs/config.yaml` to customize:
- Data sources and date ranges
- Feature engineering parameters
- Model hyperparameters
- Backtesting settings

## Next Steps

- Implement advanced features (order flow, sentiment)
- Add XGBoost and neural network models
- Integrate MLflow for experiment tracking
- Add DVC for data versioning
- Deploy with Docker

## License

MIT
