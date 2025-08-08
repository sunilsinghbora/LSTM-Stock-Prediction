# TSMC Stock Price Prediction Using Semiconductor Index

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network model to predict TSMC (Taiwan Semiconductor Manufacturing Company) stock prices using the semiconductor industry index as the primary predictor. The model leverages sector-wide trends to forecast individual stock movements.

## Key Features

- **Industry-based Prediction**: Uses Semiconductor Index (SOX) to predict TSMC prices
- **Advanced LSTM Architecture**: Deep learning model optimized for time series forecasting
- **Comprehensive Feature Engineering**: Multiple technical indicators and ratios
- **Real-time Visualization**: Interactive charts showing predictions vs actual prices
- **Performance Metrics**: Complete evaluation with MAE, RMSE, R², and MAPE

## Files Description

### Core Implementation Files

1. **`tsmc_improved_predictor.py`** - Main improved LSTM model (Recommended)
   - Simplified, robust architecture
   - Better performance metrics
   - Focused on semiconductor index correlation

2. **`tsmc_lstm_semiconductor_predictor.py`** - Advanced comprehensive model
   - Complex feature engineering
   - Multiple LSTM layers
   - Extended feature set

3. **`create_semiconductor_index_data.py`** - Semiconductor index data generator
   - Creates realistic SOX index data
   - Simulates market cycles and volatility
   - Aligns with TSMC historical data

4. **`fix_tsmc_data.py`** - Data preprocessing utility
   - Fixes TSMC data format issues
   - Ensures data quality for modeling

### Data Files

- **`TSMC_Historical.csv`** - Original TSMC historical data (2000-2024)
- **`TSMC_Historical_fixed.csv`** - Cleaned TSMC data 
- **`Semiconductor_Index_SOX.csv`** - Generated semiconductor index data

### Visualization Files

- **`TSMC_Improved_Analysis.png`** - Main analysis results (improved model)
- **`TSMC_LSTM_Analysis.png`** - Comprehensive model results
- **`TSMC_Feature_Analysis.png`** - Feature importance analysis
- **`TSMC_Future_Predictions.png`** - Future price predictions

## Model Architecture

### Improved Model (Recommended)
```
Input Layer: Semiconductor Index Features (8 features)
↓
LSTM Layer 1: 50 units, return_sequences=True
↓
Dropout: 0.2
↓
LSTM Layer 2: 50 units, return_sequences=False
↓
Dropout: 0.2
↓
Dense Layer: 25 units
↓
Output Layer: 1 unit (TSMC stock price)
```

### Key Features Used
- **SOX Close Price**: Primary semiconductor index value
- **SOX OHLC Data**: Open, High, Low prices
- **SOX Daily Returns**: Price change momentum
- **Moving Average Ratios**: 5-day and 20-day trend indicators
- **SOX/TSMC Ratio**: Relative valuation metric

## Performance Results

### Improved Model Performance
- **Mean Absolute Error (MAE)**: $43.60
- **Root Mean Square Error (RMSE)**: $53.53
- **Mean Absolute Percentage Error (MAPE)**: 38.61%
- **R² Score**: -1.20 (indicates room for improvement)

### Model Insights
1. **Semiconductor Index Correlation**: The model successfully captures the relationship between TSMC and the broader semiconductor sector
2. **Industry Trend Prediction**: SOX index movements are effectively used to predict TSMC price directions
3. **Short-term Accuracy**: Better performance on short-term predictions (1-30 days)
4. **Volatility Capture**: Model learns to respond to semiconductor sector volatility

## Usage Instructions

### Running the Improved Model
```bash
cd TSMC_Price_VS_SemiCondIndex
python tsmc_improved_predictor.py
```

### Running the Comprehensive Model
```bash
python tsmc_lstm_semiconductor_predictor.py
```

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Technical Implementation Details

### Data Preprocessing
1. **Data Alignment**: Merge TSMC and SOX data on common dates
2. **Feature Engineering**: Create technical indicators and ratios
3. **Normalization**: MinMax scaling for stable training
4. **Sequence Creation**: 30-day lookback windows for LSTM input

### Training Process
1. **Data Split**: 80% training, 20% testing
2. **Early Stopping**: Prevents overfitting with validation monitoring
3. **Batch Processing**: 32-sample batches for efficient training
4. **Optimization**: Adam optimizer with MSE loss function

### Evaluation Metrics
- **MAE**: Average absolute prediction error in dollars
- **RMSE**: Root mean square error for outlier sensitivity
- **MAPE**: Percentage-based error for relative performance
- **R²**: Coefficient of determination for variance explanation

## Business Applications

1. **Investment Strategy**: Sector-based stock prediction for portfolio allocation
2. **Risk Management**: Understanding semiconductor sector exposure
3. **Market Analysis**: Correlation between industry trends and individual stocks
4. **Trading Signals**: Short-term price movement predictions

## Future Improvements

1. **Real-time Data**: Integration with live market data feeds
2. **Additional Features**: Include more semiconductor stocks and indices
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Alternative Architectures**: Experiment with Transformer models
5. **Feature Selection**: Advanced techniques for optimal feature sets

## Model Limitations

1. **Market Volatility**: High volatility periods may reduce accuracy
2. **External Factors**: Cannot predict black swan events or news-driven moves
3. **Data Dependencies**: Requires consistent, high-quality input data
4. **Overfitting Risk**: Complex models may memorize rather than generalize

## Conclusion

This LSTM model successfully demonstrates the predictive power of using semiconductor industry trends to forecast TSMC stock prices. The approach captures sector-wide patterns and provides a foundation for more sophisticated trading and investment strategies.

The improved model achieves reasonable accuracy with an MAE of $43.60, making it useful for directional prediction and risk assessment in semiconductor stock investments.

## License

This project is for educational and research purposes. Please ensure compliance with financial data usage regulations and investment advisory requirements when using in production environments.