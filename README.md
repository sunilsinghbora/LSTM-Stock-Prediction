# Transformer-based Cross-Asset Model for TXN Return Forecasting

This project trains a transformer model on multiple large-cap stocks to forecast the next-day log return of Texas Instruments (TXN). It uses daily data from Yahoo Finance (via `yfinance`). The model ingests a window (e.g. last 60 trading days) of multivariate features (returns, rolling stats) and predicts the next day's TXN log return.

> **Disclaimer**: Educational only. Not financial advice.

## Features

- **Multivariate time series dataset construction**: Uses data from multiple large-cap stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA) to predict TXN returns
- **Engineered features**: Rolling mean/volatility, volume z-scores, and log returns across multiple time windows
- **Transformer encoder architecture**: Modern attention-based model with positional encoding and multi-layer processing
- **Early stopping and checkpointing**: Robust training with automatic model saving and validation monitoring
- **Comprehensive evaluation**: RMSE, MAE, directional accuracy, and visualization tools
- **Data leakage prevention**: Proper chronological splits and scaler fitting only on training data

## Quickstart

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Download data
python src/download_data.py --start 2015-01-01 --end 2025-01-01

# Create processed dataset
python src/features.py

# Train
python src/train.py --config config.yaml

# Evaluate
python src/evaluate.py --checkpoint checkpoints/best.pt
```

## Configuration

Edit `config.yaml` to adjust:
- **Hyperparameters**: Model architecture (d_model, attention heads, layers)
- **Tickers**: Which stocks to include for cross-asset features  
- **Window length**: How many days to look back (default: 60)
- **Training parameters**: Batch size, learning rate, early stopping patience

## Model Architecture

The PyTorch Transformer encoder model consists of:

1. **Linear projection** of features → d_model dimensions
2. **Positional encoding** to capture temporal patterns
3. **Stacked TransformerEncoder layers** with multi-head attention
4. **Mean pooling** over the sequence dimension
5. **MLP head** to predict next-day TXN log return

### Key Features:
- Input: (batch_size, sequence_length, num_features)  
- Multi-head attention with 8 heads by default
- 4 transformer layers with 1024-dim feedforward networks
- Dropout regularization (10%) to prevent overfitting
- Single output for regression (TXN next-day log return)

You can easily swap to:
- Multi-horizon decoder for predicting multiple days ahead
- Probabilistic head (e.g., output mean & variance) for uncertainty estimation

## Data Pipeline

### Data Processing Steps:
1. **Download**: Multi-asset OHLCV data via yfinance
2. **Feature Engineering**: 
   - Log returns for all assets
   - Rolling statistics (5, 10, 20-day windows)  
   - Volume z-scores for market activity signals
3. **Sequence Creation**: Sliding windows of 60 trading days
4. **Scaling**: StandardScaler fitted only on training data
5. **Splitting**: Chronological train/validation/test splits

### Data Leakage Prevention:
- ✅ Fit scalers only on train split
- ✅ Chronological splits (train/val/test) - no shuffling
- ✅ No future-aware features
- ✅ Target variable (TXN returns) excluded from feature matrix

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **MSE/RMSE/MAE**: Standard regression error metrics
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct sign predictions
- **Correlation**: Linear correlation between actual and predicted returns

### Visualization:
- Actual vs Predicted scatter plots
- Time series comparison plots  
- Residuals analysis
- Distribution comparisons
- Directional accuracy by return magnitude

## Project Structure

```
├── src/
│   ├── download_data.py    # Data fetching from Yahoo Finance
│   ├── features.py         # Feature engineering and preprocessing  
│   ├── train.py           # Transformer model training
│   └── evaluate.py        # Model evaluation and visualization
├── data/                  # Processed data files (.npy arrays)
├── checkpoints/           # Saved model checkpoints
├── results/              # Evaluation results and plots
├── config.yaml           # Configuration file
└── requirements.txt      # Python dependencies
```

## Requirements

- **PyTorch**: Deep learning framework
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Preprocessing and metrics
- **yfinance**: Stock data API
- **matplotlib/seaborn**: Visualization
- **pyyaml**: Configuration loading

## Performance

On synthetic test data (4 years of daily data):
- **RMSE**: ~0.019 (log returns)
- **MAE**: ~0.016 (log returns)  
- **Directional Accuracy**: ~41-55% (varies by market conditions)

*Note: Performance will vary significantly with real market data and different time periods*

## Extensions

Future improvements could include:

### Data:
- Add macro covariates (VIX, treasury yields, SPY)
- Include fundamental data (P/E ratios, earnings)
- Alternative data sources (news sentiment, options flow)

### Model:
- Multi-horizon forecasting (predict 1, 5, 10 days ahead)
- Quantile regression for risk estimation
- Ensemble with classical models (VAR, LightGBM)
- Cross-attention between different asset classes

### Strategy:
- Walk-forward re-training for live deployment
- Portfolio optimization using predictions
- Risk management integration

## Research Notes

This implementation demonstrates:
- Modern transformer architecture for financial time series
- Cross-asset learning (using multiple stocks to predict one)
- Proper ML practices for time series (no data leakage)
- Comprehensive evaluation beyond just accuracy

The model learns patterns across multiple assets that may be predictive of TXN's next-day returns, potentially capturing market-wide trends, sector rotations, or correlated movements.

## License

MIT License - see repository for details.