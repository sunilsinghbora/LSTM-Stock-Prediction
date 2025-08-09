#!/usr/bin/env python3
"""
Feature Engineering Module for Transformer-based Cross-Asset TXN Prediction

This module processes raw stock data to create engineered features for training
the transformer model. Features include log returns, rolling statistics,
and volume z-scores.

Usage:
    python src/features.py --config config.yaml
    python src/features.py --data-file data/stock_data_latest.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_stock_data(data_path: str) -> pd.DataFrame:
    """Load stock data from CSV file."""
    logger.info(f"Loading data from: {data_path}")
    
    # Load data with multi-level column headers
    data = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
    
    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for each stock's closing prices.
    
    Args:
        data: Stock data with multi-level columns
        
    Returns:
        DataFrame with log returns
    """
    logger.info("Calculating log returns...")
    
    returns_data = {}
    tickers = data.columns.get_level_values(0).unique()
    
    for ticker in tickers:
        if ('Close' in data[ticker].columns or 
            'Adj Close' in data[ticker].columns):
            # Use Adj Close if available, otherwise Close
            if 'Adj Close' in data[ticker].columns:
                prices = data[ticker]['Adj Close']
            else:
                prices = data[ticker]['Close']
            
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1))
            returns_data[f'{ticker}_return'] = log_returns
    
    returns_df = pd.DataFrame(returns_data, index=data.index)
    logger.info(f"Calculated log returns for {len(returns_data)} assets")
    
    return returns_df


def calculate_rolling_features(data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Calculate rolling mean and volatility features.
    
    Args:
        data: Returns data
        windows: List of rolling window sizes
        
    Returns:
        DataFrame with rolling features
    """
    logger.info(f"Calculating rolling features for windows: {windows}")
    
    rolling_features = {}
    
    for col in data.columns:
        for window in windows:
            # Rolling mean
            rolling_mean = data[col].rolling(window=window).mean()
            rolling_features[f'{col}_mean_{window}d'] = rolling_mean
            
            # Rolling volatility (standard deviation)
            rolling_vol = data[col].rolling(window=window).std()
            rolling_features[f'{col}_vol_{window}d'] = rolling_vol
    
    rolling_df = pd.DataFrame(rolling_features, index=data.index)
    logger.info(f"Created {len(rolling_features)} rolling features")
    
    return rolling_df


def calculate_volume_features(data: pd.DataFrame, zscore_window: int = 20) -> pd.DataFrame:
    """
    Calculate volume-based features including z-scores.
    
    Args:
        data: Stock data with multi-level columns
        zscore_window: Window for volume z-score calculation
        
    Returns:
        DataFrame with volume features
    """
    logger.info("Calculating volume features...")
    
    volume_features = {}
    tickers = data.columns.get_level_values(0).unique()
    
    for ticker in tickers:
        if 'Volume' in data[ticker].columns:
            volume = data[ticker]['Volume']
            
            # Volume z-score (standardized volume)
            rolling_mean = volume.rolling(window=zscore_window).mean()
            rolling_std = volume.rolling(window=zscore_window).std()
            volume_zscore = (volume - rolling_mean) / rolling_std
            
            volume_features[f'{ticker}_volume_zscore'] = volume_zscore
            
            # Log volume (to handle skewness)
            log_volume = np.log(volume + 1)  # +1 to handle zero volumes
            volume_features[f'{ticker}_log_volume'] = log_volume
    
    volume_df = pd.DataFrame(volume_features, index=data.index)
    logger.info(f"Created {len(volume_features)} volume features")
    
    return volume_df


def create_sequences(features: pd.DataFrame, target: pd.Series, 
                    window_length: int, target_ticker: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for transformer input and targets for prediction.
    
    Args:
        features: Combined feature matrix
        target: Target variable (TXN next-day returns)
        window_length: Length of input sequences
        target_ticker: Target ticker symbol
        
    Returns:
        Tuple of (X, y) arrays for model training
    """
    logger.info(f"Creating sequences with window length: {window_length}")
    
    # Remove any rows with NaN values
    valid_idx = features.dropna().index
    features_clean = features.loc[valid_idx]
    target_clean = target.loc[valid_idx]
    
    # Ensure target is properly aligned (next-day return)
    target_shifted = target_clean.shift(-1)  # Shift target forward by 1 day
    
    # Remove the last row since we don't have next-day target
    features_clean = features_clean.iloc[:-1]
    target_shifted = target_shifted.iloc[:-1]
    
    X, y = [], []
    
    for i in range(window_length, len(features_clean)):
        # Get sequence of features (window_length days)
        X_seq = features_clean.iloc[i-window_length:i].values
        
        # Get next-day target return
        y_target = target_shifted.iloc[i]
        
        X.append(X_seq)
        y.append(y_target)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, 
              train_split: float, val_split: float) -> Tuple:
    """
    Split data chronologically into train/validation/test sets.
    
    Args:
        X: Input sequences
        y: Target values
        train_split: Fraction for training (e.g., 0.7)
        val_split: Fraction for validation (e.g., 0.2)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = len(X)
    
    # Calculate split indices (chronological order)
    train_idx = int(n_samples * train_split)
    val_idx = int(n_samples * (train_split + val_split))
    
    # Split data
    X_train = X[:train_idx]
    X_val = X[train_idx:val_idx]
    X_test = X[val_idx:]
    
    y_train = y[:train_idx]
    y_val = y[train_idx:val_idx]
    y_test = y[val_idx:]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                  output_dir: str = 'data') -> Tuple:
    """
    Scale features using StandardScaler fitted only on training data.
    
    Args:
        X_train: Training sequences
        X_val: Validation sequences  
        X_test: Test sequences
        output_dir: Directory to save scaler
        
    Returns:
        Tuple of scaled (X_train, X_val, X_test) and fitted scaler
    """
    logger.info("Scaling features...")
    
    # Reshape for scaling (combine all samples and time steps)
    n_train, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    
    # Scale validation and test data
    n_val = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    
    n_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    
    # Save scaler for inference
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                       feature_names: List[str], output_dir: str = 'data') -> None:
    """Save processed data for model training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    logger.info(f"Processed data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create features for TXN prediction model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-file', type=str, 
                       help='Path to stock data CSV file')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set data file path
    data_file = args.data_file or os.path.join(args.output_dir, 'stock_data_latest.csv')
    
    # Extract config parameters
    target_ticker = config['data']['target_ticker']
    window_length = config['data']['window_length']
    use_returns = config['features']['use_returns']
    use_volume = config['features']['use_volume']
    rolling_windows = config['features']['rolling_windows']
    volume_zscore_window = config['features']['volume_zscore_window']
    val_split = config['training']['validation_split']
    test_split = config['training']['test_split']
    train_split = 1.0 - val_split - test_split
    
    try:
        # Load raw stock data
        raw_data = load_stock_data(data_file)
        
        # Calculate log returns
        returns_data = calculate_log_returns(raw_data)
        
        # Calculate rolling features
        rolling_data = calculate_rolling_features(returns_data, rolling_windows)
        
        # Calculate volume features if requested
        if use_volume:
            volume_data = calculate_volume_features(raw_data, volume_zscore_window)
            # Combine all features
            all_features = pd.concat([returns_data, rolling_data, volume_data], axis=1)
        else:
            all_features = pd.concat([returns_data, rolling_data], axis=1)
        
        # Extract target variable (TXN returns)
        target_col = f'{target_ticker}_return'
        if target_col not in all_features.columns:
            raise ValueError(f"Target column {target_col} not found in features")
        
        target = all_features[target_col]
        
        # Remove target from features to avoid data leakage
        features = all_features.drop(columns=[target_col])
        
        logger.info(f"Final feature matrix shape: {features.shape}")
        logger.info(f"Number of features: {len(features.columns)}")
        
        # Create sequences
        X, y = create_sequences(features, target, window_length, target_ticker)
        
        # Split data chronologically
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, train_split, val_split)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test, args.output_dir)
        
        # Save processed data
        save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            list(features.columns), args.output_dir)
        
        logger.info("Feature engineering completed successfully!")
        logger.info(f"Final shapes - X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()