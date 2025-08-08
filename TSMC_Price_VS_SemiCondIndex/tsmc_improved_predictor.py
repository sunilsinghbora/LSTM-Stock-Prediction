#!/usr/bin/env python3
"""
Improved TSMC Stock Price Prediction using Semiconductor Index
A more robust and simpler version focusing on core semiconductor index features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow import keras
from tensorflow.keras import layers
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class ImprovedTSMCPredictor:
    """
    Improved LSTM model for TSMC stock prediction using semiconductor index
    """
    
    def __init__(self, lookback_window=30, train_split=0.8):
        self.lookback_window = lookback_window
        self.train_split = train_split
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare data with focus on semiconductor index correlation"""
        print("Loading TSMC and Semiconductor Index data...")
        
        # Load data
        tsmc = pd.read_csv('TSMC_Historical_fixed.csv')
        sox = pd.read_csv('Semiconductor_Index_SOX.csv')
        
        # Convert dates and sort
        tsmc['Date'] = pd.to_datetime(tsmc['Date'])
        sox['Date'] = pd.to_datetime(sox['Date'])
        
        # Merge on date (inner join to get common dates only)
        data = pd.merge(tsmc, sox, on='Date', suffixes=('_TSMC', '_SOX'), how='inner')
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"Merged data shape: {data.shape}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Create focused features
        # Focus on semiconductor index as primary predictor
        data['SOX_Return'] = data['Close_SOX'].pct_change()
        data['TSMC_Return'] = data['Close_TSMC'].pct_change()
        data['SOX_TSMC_Ratio'] = data['Close_SOX'] / data['Close_TSMC']
        data['SOX_Volume_Norm'] = (data['Volume_SOX'] - data['Volume_SOX'].rolling(20).mean()) / data['Volume_SOX'].rolling(20).std()
        
        # Moving averages for trend
        for window in [5, 20]:
            data[f'SOX_MA_{window}'] = data['Close_SOX'].rolling(window).mean()
            data[f'SOX_MA_{window}_Ratio'] = data['Close_SOX'] / data[f'SOX_MA_{window}']
        
        # Select key features - semiconductor index focused
        feature_cols = [
            'Close_SOX',           # Primary: SOX closing price
            'Open_SOX',            # SOX opening price
            'High_SOX',            # SOX high
            'Low_SOX',             # SOX low
            'SOX_Return',          # SOX daily return
            'SOX_MA_5_Ratio',      # SOX vs 5-day MA
            'SOX_MA_20_Ratio',     # SOX vs 20-day MA
            'SOX_TSMC_Ratio',      # Relative valuation
        ]
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        # Select features and target
        X = data[feature_cols].values
        y = data['Close_TSMC'].values  # Predict TSMC closing price
        dates = data['Date'].values
        
        print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        return X, y, dates
    
    def prepare_sequences(self, X, y, dates):
        """Create LSTM sequences"""
        print(f"Creating sequences with lookback window: {self.lookback_window}")
        
        # Create sequences
        X_seq, y_seq, date_seq = [], [], []
        
        for i in range(self.lookback_window, len(X)):
            X_seq.append(X[i-self.lookback_window:i])
            y_seq.append(y[i])
            date_seq.append(dates[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        date_seq = np.array(date_seq)
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
        X_scaled = X_scaled.reshape(X_seq.shape)
        
        y_scaled = self.scaler_y.fit_transform(y_seq.reshape(-1, 1)).flatten()
        
        # Split data
        split_idx = int(len(X_scaled) * self.train_split)
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_test = y_scaled[split_idx:]
        
        self.test_dates = date_seq[split_idx:]
        self.test_actual = y_seq[split_idx:]
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build a simpler, more robust LSTM model"""
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline"""
        # Load data
        X, y, dates = self.load_and_prepare_data()
        
        # Prepare sequences
        X_train, X_test, y_train, y_test = self.prepare_sequences(X, y, dates)
        
        # Build and train model
        print("\nBuilding and training LSTM model...")
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(self.test_actual, y_pred)
        mae = mean_absolute_error(self.test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.test_actual, y_pred)
        mape = np.mean(np.abs((self.test_actual - y_pred.flatten()) / self.test_actual)) * 100
        
        print(f"\nModel Performance:")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Plot results
        self.plot_results(y_pred, history)
        
        return {
            'predictions': y_pred,
            'actual': self.test_actual,
            'dates': self.test_dates,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
            'history': history
        }
    
    def plot_results(self, y_pred, history):
        """Create comprehensive result visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Actual vs Predicted
        axes[0,0].plot(self.test_dates, self.test_actual, label='Actual TSMC Price', color='blue', linewidth=2)
        axes[0,0].plot(self.test_dates, y_pred.flatten(), label='Predicted Price', color='red', linewidth=2, alpha=0.8)
        axes[0,0].set_title('TSMC Stock Price Prediction using Semiconductor Index', fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Price ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Format dates
        axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Training History
        axes[0,1].plot(history.history['loss'], label='Training Loss')
        axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0,1].set_title('Training History', fontweight='bold')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot
        axes[1,0].scatter(self.test_actual, y_pred, alpha=0.6, color='green')
        min_price = min(self.test_actual.min(), y_pred.min())
        max_price = max(self.test_actual.max(), y_pred.max())
        axes[1,0].plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2)
        axes[1,0].set_title('Actual vs Predicted Scatter', fontweight='bold')
        axes[1,0].set_xlabel('Actual Price')
        axes[1,0].set_ylabel('Predicted Price')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Recent performance (last 100 days)
        recent_idx = -min(100, len(self.test_dates))
        axes[1,1].plot(self.test_dates[recent_idx:], self.test_actual[recent_idx:], 
                      label='Actual', color='blue', linewidth=2)
        axes[1,1].plot(self.test_dates[recent_idx:], y_pred[recent_idx:], 
                      label='Predicted', color='red', linewidth=2)
        axes[1,1].set_title('Recent Performance (Last 100 Days)', fontweight='bold')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Price ($)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('TSMC_Improved_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the improved TSMC prediction analysis"""
    print("="*70)
    print("IMPROVED TSMC STOCK PREDICTION USING SEMICONDUCTOR INDEX")
    print("="*70)
    
    predictor = ImprovedTSMCPredictor(lookback_window=30, train_split=0.8)
    results = predictor.train_and_evaluate()
    
    print("\nModel successfully trained!")
    print("Key insight: The model uses semiconductor industry trends (SOX index)")
    print("to predict TSMC stock movements, capturing sector-wide patterns.")
    print("\nVisualization saved as: TSMC_Improved_Analysis.png")
    
    return results

if __name__ == "__main__":
    results = main()