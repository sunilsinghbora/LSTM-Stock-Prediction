#!/usr/bin/env python3
"""
Fix TSMC data format issues
"""

import pandas as pd
import numpy as np

def fix_tsmc_data():
    """
    Fix the TSMC data format where Volume is stored in Adj Close column
    """
    print("Fixing TSMC data format...")
    
    # Load original data
    tsmc_data = pd.read_csv('TSMC_Historical.csv')
    
    # The issue is that Volume is in Adj Close column and Volume column is NaN
    # Let's fix this by moving the values
    tsmc_data['Volume'] = tsmc_data['Adj Close']
    tsmc_data['Adj Close'] = tsmc_data['Close']  # Use Close price as Adj Close for simplicity
    
    # Save fixed data
    tsmc_data.to_csv('TSMC_Historical_fixed.csv', index=False)
    
    print("TSMC data fixed and saved as TSMC_Historical_fixed.csv")
    print("Shape:", tsmc_data.shape)
    print("Columns:", list(tsmc_data.columns))
    print("\nFirst few rows:")
    print(tsmc_data.head())
    print("\nVolume statistics:")
    print(tsmc_data['Volume'].describe())
    
    return tsmc_data

if __name__ == "__main__":
    fix_tsmc_data()