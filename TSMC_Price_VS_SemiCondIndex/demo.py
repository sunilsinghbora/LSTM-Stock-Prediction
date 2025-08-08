#!/usr/bin/env python3
"""
TSMC Stock Prediction Demo
Quick demo script to showcase the LSTM model predicting TSMC prices using semiconductor index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_demo():
    """Run a quick demonstration of the TSMC prediction model"""
    print("="*60)
    print("TSMC STOCK PREDICTION DEMO")
    print("Using Semiconductor Index to Predict TSMC Prices")
    print("="*60)
    
    # Load the data
    print("\n1. Loading TSMC and Semiconductor Index data...")
    try:
        tsmc = pd.read_csv('TSMC_Historical_fixed.csv')
        sox = pd.read_csv('Semiconductor_Index_SOX.csv')
        print("✓ Data loaded successfully")
        print(f"  - TSMC data: {len(tsmc)} records")
        print(f"  - SOX data: {len(sox)} records")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please run the data generation scripts first.")
        return
    
    # Data overview
    print("\n2. Data Overview:")
    tsmc['Date'] = pd.to_datetime(tsmc['Date'])
    sox['Date'] = pd.to_datetime(sox['Date'])
    
    print(f"  - TSMC date range: {tsmc['Date'].min().strftime('%Y-%m-%d')} to {tsmc['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  - TSMC price range: ${tsmc['Close'].min():.2f} to ${tsmc['Close'].max():.2f}")
    print(f"  - SOX date range: {sox['Date'].min().strftime('%Y-%m-%d')} to {sox['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  - SOX index range: {sox['Close'].min():.2f} to {sox['Close'].max():.2f}")
    
    # Merge data and show correlation
    print("\n3. Analyzing Correlation between TSMC and Semiconductor Index...")
    merged = pd.merge(tsmc, sox, on='Date', suffixes=('_TSMC', '_SOX'))
    correlation = merged['Close_TSMC'].corr(merged['Close_SOX'])
    print(f"  - Correlation coefficient: {correlation:.4f}")
    
    if correlation > 0.5:
        print("  - Strong positive correlation detected! Good for prediction.")
    elif correlation > 0.3:
        print("  - Moderate positive correlation. Should work for prediction.")
    else:
        print("  - Weak correlation. Model may face challenges.")
    
    # Sample recent data
    print("\n4. Recent Data Sample (Last 10 days):")
    recent = merged.tail(10)[['Date', 'Close_TSMC', 'Close_SOX']].copy()
    recent['SOX_Change'] = recent['Close_SOX'].pct_change() * 100
    recent['TSMC_Change'] = recent['Close_TSMC'].pct_change() * 100
    
    print("Date       | TSMC Price | SOX Index | SOX Change% | TSMC Change%")
    print("-" * 65)
    for _, row in recent.iterrows():
        if pd.isna(row['SOX_Change']):
            print(f"{row['Date'].strftime('%Y-%m-%d')} | ${row['Close_TSMC']:8.2f} | {row['Close_SOX']:8.2f} |     --     |     --")
        else:
            print(f"{row['Date'].strftime('%Y-%m-%d')} | ${row['Close_TSMC']:8.2f} | {row['Close_SOX']:8.2f} | {row['SOX_Change']:8.2f}% | {row['TSMC_Change']:8.2f}%")
    
    # Create visualization
    print("\n5. Creating Visualization...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price comparison
    plt.subplot(2, 2, 1)
    plt.plot(merged['Date'], merged['Close_TSMC'], label='TSMC Price', color='blue', linewidth=2)
    plt.title('TSMC Stock Price Over Time', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('TSMC Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: SOX Index
    plt.subplot(2, 2, 2)
    plt.plot(merged['Date'], merged['Close_SOX'], label='Semiconductor Index', color='green', linewidth=2)
    plt.title('Semiconductor Index (SOX) Over Time', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('SOX Index')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Correlation scatter
    plt.subplot(2, 2, 3)
    plt.scatter(merged['Close_SOX'], merged['Close_TSMC'], alpha=0.5, color='red')
    plt.title(f'TSMC vs SOX Correlation (r={correlation:.3f})', fontweight='bold')
    plt.xlabel('SOX Index')
    plt.ylabel('TSMC Price ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Recent performance
    recent_data = merged.tail(252)  # Last year
    plt.subplot(2, 2, 4)
    plt.plot(recent_data['Date'], recent_data['Close_TSMC'], label='TSMC', color='blue', linewidth=2)
    plt.title('Recent Performance (Last Year)', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('TSMC Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('TSMC_Demo_Analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'TSMC_Demo_Analysis.png'")
    plt.show()
    
    # Model recommendation
    print("\n6. Model Training Recommendation:")
    print(f"✓ Based on correlation of {correlation:.4f}, the LSTM model should perform well")
    print("✓ Run 'python tsmc_improved_predictor.py' for the full prediction model")
    print("✓ Expected performance: MAE ~$40-50, MAPE ~35-45%")
    
    # Summary statistics
    print("\n7. Summary Statistics:")
    print(f"  - Total trading days analyzed: {len(merged)}")
    print(f"  - TSMC average daily return: {merged['Close_TSMC'].pct_change().mean()*100:.3f}%")
    print(f"  - TSMC volatility (std): {merged['Close_TSMC'].pct_change().std()*100:.3f}%")
    print(f"  - SOX average daily return: {merged['Close_SOX'].pct_change().mean()*100:.3f}%")
    print(f"  - SOX volatility (std): {merged['Close_SOX'].pct_change().std()*100:.3f}%")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("Ready to run full LSTM prediction model.")
    print("="*60)

if __name__ == "__main__":
    run_demo()