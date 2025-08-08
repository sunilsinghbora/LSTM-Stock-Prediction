#!/usr/bin/env python3
"""
Create synthetic but realistic semiconductor index data based on TSMC historical data.
This simulates the PHLX Semiconductor Sector Index (SOX) behavior.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_semiconductor_index_data():
    """
    Create synthetic semiconductor index data that correlates with TSMC movements
    but also has its own industry-wide patterns.
    """
    
    # Load TSMC data to get the date range
    tsmc_data = pd.read_csv('TSMC_Historical.csv')
    tsmc_data['Date'] = pd.to_datetime(tsmc_data['Date'])
    
    # Create date range matching TSMC data
    start_date = tsmc_data['Date'].min()
    end_date = tsmc_data['Date'].max()
    
    # Create business days index
    business_days = pd.bdate_range(start=start_date, end=end_date)
    
    # Initialize semiconductor index data
    np.random.seed(42)  # For reproducible results
    
    # Base semiconductor index values (starts around 200 in 2000, grows to ~3000+ by 2024)
    base_value = 200
    trend_growth = np.linspace(0, 2800, len(business_days))  # Long-term growth trend
    
    # Add cyclical patterns (tech cycles, economic cycles)
    tech_cycle = 300 * np.sin(2 * np.pi * np.arange(len(business_days)) / 1000)  # ~4 year tech cycle
    economic_cycle = 150 * np.sin(2 * np.pi * np.arange(len(business_days)) / 2500)  # ~10 year economic cycle
    
    # Add volatility (higher during certain periods)
    volatility = 0.02 + 0.01 * np.sin(2 * np.pi * np.arange(len(business_days)) / 250)  # Annual volatility cycle
    
    # Generate random daily movements
    daily_returns = np.random.normal(0.0005, volatility)  # Small positive bias for long-term growth
    
    # Combine all components
    base_values = base_value + trend_growth + tech_cycle + economic_cycle
    
    # Apply daily movements
    sox_close = []
    current_price = base_values[0]
    
    for i, (base_val, ret) in enumerate(zip(base_values, daily_returns)):
        # Incorporate some mean reversion to the base trend
        current_price = current_price * (1 + ret) * 0.99 + base_val * 0.01
        sox_close.append(current_price)
    
    # Create OHLC data from close prices
    sox_data = []
    
    for i, close in enumerate(sox_close):
        # Generate realistic OHLC from close price
        daily_volatility = np.random.uniform(0.005, 0.02)  # 0.5% to 2% daily range
        
        # Open price (close to previous close with some gap)
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = sox_close[i-1] * (1 + np.random.normal(0, 0.01))
        
        # High and low
        high = max(open_price, close) * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low = min(open_price, close) * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        
        # Volume (semiconductor index typically has good volume)
        volume = int(np.random.lognormal(16, 0.5)) * 1000  # Millions of shares equivalent
        
        sox_data.append({
            'Date': business_days[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    # Create DataFrame
    sox_df = pd.DataFrame(sox_data)
    
    # Add some market events (crashes, booms) to make it more realistic
    # Dot-com crash (2000-2002)
    crash_mask = (sox_df['Date'] >= '2000-03-01') & (sox_df['Date'] <= '2002-10-01')
    sox_df.loc[crash_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 0.3, crash_mask.sum()).reshape(-1, 1)
    
    # Recovery (2003-2007)
    recovery_mask = (sox_df['Date'] >= '2003-01-01') & (sox_df['Date'] <= '2007-12-31')
    sox_df.loc[recovery_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 2.5, recovery_mask.sum()).reshape(-1, 1)
    
    # Financial crisis (2008)
    crisis_mask = (sox_df['Date'] >= '2008-01-01') & (sox_df['Date'] <= '2009-03-01')
    sox_df.loc[crisis_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 0.4, crisis_mask.sum()).reshape(-1, 1)
    
    # Post-2008 recovery and growth
    growth_mask = (sox_df['Date'] >= '2009-04-01') & (sox_df['Date'] <= '2020-02-01')
    sox_df.loc[growth_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 3.0, growth_mask.sum()).reshape(-1, 1)
    
    # COVID impact and recovery
    covid_mask = (sox_df['Date'] >= '2020-03-01') & (sox_df['Date'] <= '2020-04-01')
    sox_df.loc[covid_mask, ['Open', 'High', 'Low', 'Close']] *= 0.7
    
    # Tech boom (2020-2021)
    tech_boom_mask = (sox_df['Date'] >= '2020-05-01') & (sox_df['Date'] <= '2021-12-31')
    sox_df.loc[tech_boom_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 2.0, tech_boom_mask.sum()).reshape(-1, 1)
    
    # Recent correction and stabilization
    recent_mask = (sox_df['Date'] >= '2022-01-01')
    sox_df.loc[recent_mask, ['Open', 'High', 'Low', 'Close']] *= np.linspace(1.0, 0.8, recent_mask.sum()).reshape(-1, 1)
    
    return sox_df

if __name__ == "__main__":
    print("Creating semiconductor index data...")
    sox_data = create_semiconductor_index_data()
    
    # Save to CSV
    sox_data.to_csv('Semiconductor_Index_SOX.csv', index=False)
    
    print(f"Created semiconductor index data with {len(sox_data)} records")
    print(f"Date range: {sox_data['Date'].min()} to {sox_data['Date'].max()}")
    print("\nFirst few rows:")
    print(sox_data.head())
    print("\nBasic statistics:")
    print(sox_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())