#!/usr/bin/env python3

"""
TSMC Stock Price Prediction using LSTM Neural Networks

This script implements a complete LSTM model to predict TSMC stock prices
based on historical stock data. The implementation includes:

1. Data loading and exploration
2. Data visualization and analysis
3. Data preprocessing for LSTM
4. LSTM model building and training
5. Prediction and evaluation
6. Results visualization

Author: Generated for TSMC stock prediction task
Date: 2025
"""

# Import required libraries
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime

def load_and_explore_data():
    """Load TSMC historical data and perform initial exploration"""
    print("=== TSMC Stock Price Prediction with LSTM ===")
    print("\nStep 1: Loading and exploring data...")
    
    data = pd.read_csv("TSMC_Historical.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"Data loaded successfully! Shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Data info:")
    print(data.info())
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nBasic statistics:")
    print(data.describe())
    
    return data

def create_visualizations(data):
    """Create various visualizations of the stock data"""
    print("\nStep 2: Creating visualizations...")
    
    # Plot 1 - Open and Close Prices over time
    plt.figure(figsize=(15,6))
    plt.plot(data['Date'], data['Open'], label="Open", color="blue")
    plt.plot(data['Date'], data['Close'], label="Close", color="red")
    plt.title("TSMC Open-Close Price over Time")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsmc_price_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Price history plot saved")

    # Plot 2 - Trading Volume (use Adj Close since Volume is empty)
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Adj Close'], label="Adjusted Close", color="green")
    plt.title("TSMC Adjusted Close Price over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsmc_adj_close.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Adjusted close plot saved")

    # Plot 3 - Correlation heatmap
    numeric_data = data.select_dtypes(include=["int64","float64"])
    numeric_data = numeric_data.drop('Volume', axis=1, errors='ignore')

    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("TSMC Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('tsmc_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Correlation heatmap saved")

def prepare_lstm_data(data, sequence_length=50, train_split=0.9):
    """Prepare data for LSTM model training"""
    print(f"\nStep 3: Preparing data for LSTM model...")
    
    # Use Close price for prediction
    stock_close = data.filter(["Close"])
    dataset = stock_close.values
    print(f"Dataset shape: {dataset.shape}")
    
    # Split into training and testing
    training_data_len = int(np.ceil(len(dataset) * train_split))
    print(f"Training data length: {training_data_len}")
    print(f"Test data length: {len(dataset) - training_data_len}")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    
    # Prepare training data
    training_data = scaled_data[:training_data_len]
    
    # Create sequences
    X_train, y_train = [], []
    
    print(f"Creating sequences with window size: {sequence_length}")
    for i in range(sequence_length, len(training_data)):
        X_train.append(training_data[i-sequence_length:i, 0])
        y_train.append(training_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(f"✓ Training data shape: {X_train.shape}")
    print(f"✓ Training labels shape: {y_train.shape}")
    
    return X_train, y_train, scaler, scaled_data, training_data_len, sequence_length

def build_lstm_model(input_shape):
    """Build and compile the LSTM model"""
    print(f"\nStep 4: Building LSTM model...")
    
    model = keras.models.Sequential([
        # First LSTM layer with return sequences
        keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        
        # Second LSTM layer
        keras.layers.LSTM(64, return_sequences=False),
        
        # Dense layer
        keras.layers.Dense(128, activation="relu"),
        
        # Dropout for regularization
        keras.layers.Dropout(0.50),
        
        # Output layer
        keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
    
    print("✓ Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    """Train the LSTM model"""
    print(f"\nStep 5: Training the model...")
    print("This may take a few minutes...")
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1
    )
    
    print("✓ Training completed!")
    return history

def make_predictions(model, data, scaler, scaled_data, training_data_len, sequence_length):
    """Prepare test data and make predictions"""
    print(f"\nStep 6: Preparing test data and making predictions...")
    
    # Create test data
    test_data = scaled_data[training_data_len - sequence_length:]
    X_test, y_test = [], data.iloc[training_data_len:]['Close'].values
    
    for i in range(sequence_length, len(test_data)):
        X_test.append(test_data[i-sequence_length:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    print(f"✓ Test data shape: {X_test.shape}")
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    print("✓ Predictions completed!")
    
    return predictions, X_test, y_test

def evaluate_model(actual_prices, predictions):
    """Calculate and display model performance metrics"""
    
    # Handle length mismatch
    min_length = min(len(actual_prices), len(predictions))
    actual = actual_prices[:min_length]
    pred = predictions[:min_length].flatten()
    
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    print(f"\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return mae, rmse, mape

def create_prediction_visualization(data, training_data_len, predictions, history):
    """Create visualizations of predictions and training history"""
    print(f"\nStep 7: Creating prediction visualization...")
    
    # Create dataframes for plotting
    train = data[:training_data_len]
    test = data[training_data_len:].copy()
    
    # Handle length mismatch
    if len(predictions) == len(test):
        test['Predictions'] = predictions
    else:
        test = test.iloc[:len(predictions)].copy()
        test['Predictions'] = predictions
    
    # Create the final prediction plot
    plt.figure(figsize=(15,8))
    plt.plot(train['Date'], train['Close'], label="Training Data (Actual)", color='blue')
    plt.plot(test['Date'], test['Close'], label="Test Data (Actual)", color='orange')
    plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
    plt.title("TSMC Stock Price Prediction using LSTM")
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsmc_lstm_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Prediction plot saved as 'tsmc_lstm_predictions.png'")
    
    # Plot training history
    print("\nStep 8: Plotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    if 'val_root_mean_squared_error' in history.history:
        plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tsmc_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Training history plot saved")

def main():
    """Main function to run the complete TSMC stock prediction pipeline"""
    
    # Step 1: Load and explore data
    data = load_and_explore_data()
    
    # Step 2: Create visualizations
    create_visualizations(data)
    
    # Step 3: Prepare data for LSTM
    X_train, y_train, scaler, scaled_data, training_data_len, sequence_length = prepare_lstm_data(data)
    
    # Step 4: Build LSTM model
    input_shape = (X_train.shape[1], 1)
    model = build_lstm_model(input_shape)
    
    # Step 5: Train the model
    history = train_model(model, X_train, y_train)
    
    # Step 6: Make predictions
    predictions, X_test, y_test = make_predictions(model, data, scaler, scaled_data, training_data_len, sequence_length)
    
    # Step 7: Evaluate model
    evaluate_model(y_test, predictions)
    
    # Step 8: Create visualizations
    create_prediction_visualization(data, training_data_len, predictions, history)
    
    # Save model
    model.save('tsmc_lstm_model.keras')
    print("✓ Model saved as 'tsmc_lstm_model.keras'")
    
    print(f"\n=== TSMC LSTM Stock Prediction Complete! ===")
    print("Generated files:")
    print("- tsmc_price_history.png: Historical price chart")
    print("- tsmc_adj_close.png: Adjusted close price chart")
    print("- tsmc_correlation.png: Feature correlation heatmap")
    print("- tsmc_lstm_predictions.png: LSTM predictions vs actual prices")
    print("- tsmc_training_history.png: Training loss and RMSE history")
    print("- tsmc_lstm_model.keras: Saved trained model")
    print("\n✅ Implementation completed successfully!")

if __name__ == "__main__":
    main()