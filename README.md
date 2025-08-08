# LSTM-Stock-Prediction

Using LSTM (Long Short-Term Memory) neural networks to predict stock prices for different companies.

## Overview

This repository contains implementations of LSTM models for stock price prediction:

1. **Microsoft Stock Prediction** (`MicsosoftStockPrice/`) - Complete LSTM implementation for MSFT stock
2. **TSMC Stock Prediction** (`TSMC_Price_VS_SemiCondIndex/`) - Complete LSTM implementation for TSMC stock

## Features

- **Data Loading and Preprocessing**: Load historical stock data and prepare it for LSTM training
- **LSTM Model Architecture**: Deep learning model with multiple LSTM layers, dropout regularization
- **Training and Validation**: Model training with validation split and performance metrics
- **Prediction and Visualization**: Generate predictions and create comprehensive visualizations
- **Model Persistence**: Save trained models for future use

## TSMC Implementation

The TSMC stock prediction model includes:

### Key Components:
- **Data Analysis**: Historical price data from 2000-2024 (25+ years)
- **Feature Engineering**: Uses Close prices with 50-day sequence windows
- **LSTM Architecture**: 
  - 2 LSTM layers (64 units each)
  - Dense layer (128 units) with ReLU activation
  - Dropout (50%) for regularization
  - Single output layer for price prediction
- **Performance**: ~7.37% MAPE (Mean Absolute Percentage Error)

### Generated Outputs:
- `tsmc_price_history.png`: Historical price trends
- `tsmc_lstm_predictions.png`: Model predictions vs actual prices
- `tsmc_training_history.png`: Training loss and RMSE progression
- `tsmc_correlation.png`: Feature correlation heatmap
- `tsmc_lstm_model.keras`: Trained model file

## Requirements

```
tensorflow>=2.8.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.1.63
```

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run TSMC Prediction
```bash
cd TSMC_Price_VS_SemiCondIndex/
python TSMC_LSTM_Complete.py
```

### Run Microsoft Prediction
```bash
cd MicsosoftStockPrice/
python PredictPrice.py
```

## Model Performance

The TSMC LSTM model achieves:
- **Mean Absolute Error (MAE)**: ~$10.23
- **Root Mean Squared Error (RMSE)**: ~$14.30  
- **Mean Absolute Percentage Error (MAPE)**: ~7.37%

## Architecture Details

The LSTM model uses:
- **Input**: 50-day sequences of normalized stock prices
- **Hidden Layers**: 2 LSTM layers (64 units each)
- **Regularization**: 50% dropout to prevent overfitting
- **Output**: Single price prediction
- **Training**: 90/10 train/test split, Adam optimizer, MAE loss

## Data Source

- TSMC historical stock data (2000-2024)
- Microsoft historical stock data (1987-2022)

## License

MIT License - see repository for details
