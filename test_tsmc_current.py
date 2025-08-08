#!/usr/bin/env python3

# Test the current TSMC implementation to see what works and what doesn't

import os
import sys
sys.path.append('/home/runner/work/LSTM-Stock-Prediction/LSTM-Stock-Prediction')

# Change to TSMC directory
os.chdir('/home/runner/work/LSTM-Stock-Prediction/LSTM-Stock-Prediction/TSMC_Price_VS_SemiCondIndex')

# Import required libraries
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime

print("Testing current TSMC implementation...")

# Load and examine data
data = pd.read_csv("TSMC_Historical.csv")
print("Data loaded successfully!")
print("Data shape:", data.shape)
print("Data head:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nData description:")
print(data.describe())

# Test if the existing code works up to the visualization part
print("\n=== Testing visualizations ===")

# Test plotting (save to file instead of displaying)
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Open'], label="Open",color="blue")
plt.plot(data['Date'], data['Close'], label="Close",color="red")
plt.title("Open-Close Price over Time")
plt.legend()
plt.savefig('test_plot1.png')
plt.close()
print("Plot 1 saved successfully")

# Check if the notebook implementation stops here or has more
print("\n=== Checking if there's more code ===")
print("This is where the current TSMC notebook appears to end.")
print("Need to implement LSTM model similar to Microsoft example.")