#!/usr/bin/env python3
"""
Data Download Module for Transformer-based Cross-Asset TXN Prediction

This module downloads historical stock data for multiple large-cap stocks
and Texas Instruments (TXN) from Yahoo Finance. The data is used to train
a transformer model that predicts next-day TXN log returns.

Usage:
    python src/download_data.py --start 2015-01-01 --end 2025-01-01
    python src/download_data.py --config config.yaml
"""

import argparse
import os
import pandas as pd
import yfinance as yf
import yaml
from datetime import datetime
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download stock data for multiple tickers from Yahoo Finance.
    
    Args:
        tickers: List of stock symbols to download
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with multi-level columns (ticker, OHLCV)
    """
    logger.info(f"Downloading data for {len(tickers)} tickers: {tickers}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Handle single ticker case (yfinance returns different structure)
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])
    
    logger.info(f"Downloaded {len(data)} trading days of data")
    logger.info(f"Data shape: {data.shape}")
    
    return data


def validate_data(data: pd.DataFrame, tickers: List[str]) -> None:
    """
    Validate downloaded data for completeness and quality.
    
    Args:
        data: Downloaded stock data
        tickers: List of expected tickers
    """
    logger.info("Validating downloaded data...")
    
    # Check for missing tickers
    available_tickers = data.columns.get_level_values(0).unique().tolist()
    missing_tickers = set(tickers) - set(available_tickers)
    if missing_tickers:
        logger.warning(f"Missing data for tickers: {missing_tickers}")
    
    # Check data completeness for each ticker
    for ticker in available_tickers:
        ticker_data = data[ticker]
        missing_ratio = ticker_data.isnull().sum().sum() / (len(ticker_data) * len(ticker_data.columns))
        logger.info(f"{ticker}: {missing_ratio:.2%} missing data")
        
        if missing_ratio > 0.1:  # More than 10% missing
            logger.warning(f"{ticker} has high missing data ratio: {missing_ratio:.2%}")


def save_data(data: pd.DataFrame, output_dir: str = 'data') -> str:
    """
    Save downloaded data to CSV file.
    
    Args:
        data: Stock data to save
        output_dir: Directory to save data
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_data_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save with multi-level column headers
    data.to_csv(filepath)
    logger.info(f"Data saved to: {filepath}")
    
    # Also save a 'latest' version for easy access
    latest_filepath = os.path.join(output_dir, "stock_data_latest.csv")
    data.to_csv(latest_filepath)
    logger.info(f"Latest data saved to: {latest_filepath}")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Download stock data for TXN prediction model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--start', type=str, 
                       help='Start date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for data files')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        tickers = config['data']['tickers']
        start_date = args.start or config['data']['start_date']
        end_date = args.end or config['data']['end_date']
    else:
        # Default configuration if config file doesn't exist
        logger.warning(f"Config file {args.config} not found, using defaults")
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'TXN']
        start_date = args.start or '2015-01-01'
        end_date = args.end or '2025-01-01'
    
    try:
        # Download data
        data = download_stock_data(tickers, start_date, end_date)
        
        # Validate data
        validate_data(data, tickers)
        
        # Save data
        filepath = save_data(data, args.output_dir)
        
        logger.info("Data download completed successfully!")
        logger.info(f"Final dataset shape: {data.shape}")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


if __name__ == "__main__":
    main()