#!/usr/bin/env python
# coding: utf-8

# # Microsoft Stock Price Prediction Analysis - Complete Documentation
# 
# ## Project Overview
# This notebook demonstrates end-to-end stock price prediction using Long Short-Term Memory (LSTM) neural networks. We analyze 35+ years of Microsoft stock data to build a model that can forecast future prices based on historical patterns.
# 
# ### Key Learning Objectives:
# 1. **Time Series Analysis**: Understanding stock market data patterns
# 2. **Data Preprocessing**: Scaling and sequence preparation for neural networks
# 3. **LSTM Architecture**: Building deep learning models for temporal data
# 4. **Model Evaluation**: Assessing prediction accuracy and visualization
# 
# ### Business Context:
# Stock price prediction is crucial for:
# - **Investment Decisions**: Helping investors make informed choices
# - **Risk Management**: Understanding potential price movements
# - **Portfolio Optimization**: Timing buy/sell decisions
# - **Market Analysis**: Identifying trends and patterns
# 
# ---
# 
# ## Step 1: Environment Setup and Library Imports
# 
# ### Purpose:
# Import all necessary libraries for data manipulation, visualization, and machine learning operations.
# 
# ### Libraries Explained:
# 

# In[2]:


#imports
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


# ### Why These Libraries?
# - **TensorFlow/Keras**: Industry-standard for building LSTM models
# - **Pandas**: Essential for CSV data handling and time-series operations
# - **NumPy**: Efficient numerical operations required by neural networks
# - **StandardScaler**: Normalizes data to improve model training stability
# - **Matplotlib/Seaborn**: Creates professional visualizations for analysis
# - **DateTime**: Handles time-series data filtering and manipulation
# 
# ---
# 
# ## Step 2: Data Loading and Initial Exploration
# 
# ### Purpose:
# Load the Microsoft stock dataset and perform initial data quality checks to understand the structure and characteristics of our data.
# 
# ### What We're Looking For:
# - **Data Quality**: Missing values, data types, date ranges
# - **Feature Understanding**: Available columns and their meanings
# - **Statistical Overview**: Price ranges, trends, and distributions

# In[3]:


#Read the dataset
df = pd.read_csv('MSFT_shareprice1987-2022.csv')
#convert Date column to datetime and update the dataframe
df['Date'] = pd.to_datetime(df['Date'])
print(df.head())
print(df.info())
print(df.describe())


# ### Data Exploration Results:
# - **Time Period**: 1987-2022 (35+ years of historical data)
# - **Features Available**: Date, Open, High, Low, Close, Volume, Adj Close
# - **Data Points**: Thousands of daily trading records
# - **Target Variable**: Close price (what we want to predict)
# 
# ---
# 
# ## Step 3: Data Visualization and Pattern Analysis
# 
# ### 3.1 Price Trends Over Time
# Understanding long-term market behavior and identifying key patterns in Microsoft's stock performance.

# In[4]:


# Initial Data Visualization
# Plot 1 - Open and Close Prices over time
plt.figure(figsize=(24,6))
plt.plot(df['Date'], df['Open'], label="Open", color="blue")
plt.plot(df['Date'], df['Close'], label="Close", color="red")
plt.title("MSFT Open-Close Price over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Step 3: Data Visualization and Analysis
# 
# ### 3.1 Stock Price Trends Over Time
# We'll create visualizations to understand the data patterns:
# - **Open vs Close Prices**: Compare opening and closing prices over the entire time period
# - **Long-term trends**: Identify overall market trends and growth patterns
# - **Volatility analysis**: Observe periods of high and low volatility

# In[5]:


# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(df['Date'],df['Volume'],label="Volume",color="orange")
plt.title("Stock Volume over Time")
plt.show()


# ### 3.2 Trading Volume Analysis
# Understanding trading volume patterns:
# - **Volume trends**: Analyze trading volume over time
# - **Outlier detection**: Identify unusual trading activity
# - **Volume-price relationship**: Understanding how volume correlates with price movements

# In[6]:


# Drop non-numeric columns
numeric_data = df.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# ### 3.3 Feature Correlation Analysis
# Analyzing relationships between different stock metrics:
# - **Correlation heatmap**: Visualize how different features correlate with each other
# - **Feature selection**: Identify which features are most important for prediction
# - **Multicollinearity detection**: Check if features are too similar to each other

# In[7]:


# Create a date filter for the prediction period (2013-2018)
prediction = df.loc[
    (df['Date'] > datetime(2013,1,1)) &
    (df['Date'] < datetime(2018,1,1))
]

print(f"Original dataset size: {len(df)} rows")
print(f"Filtered dataset size: {len(prediction)} rows")

# Visualize the filtered data
plt.figure(figsize=(12,6))
plt.plot(prediction['Date'], prediction['Close'], color="blue", linewidth=1.5)
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.title("MSFT Close Price (2013-2018) - Prediction Period")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[29]:


# Prepare for the LSTM Model (Sequential)
stock_close = df.filter(["Close"])
dataset = stock_close.values #convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.90))

# Preprocessing Stages

# Preprocessing Stages - Data normalization and preparation for LSTM model
# Create a StandardScaler object to normalize data (mean=0, std=1)
scaler = StandardScaler()

scaled_data = scaler.fit_transform(dataset)  # Fit scaler to data and transform values to standard normal distribution

# Split the scaled data into training portion (90% of total dataset)
training_data = scaled_data[:training_data_len]  # Use first 60% of data for training the model

# Initialize empty lists to store training features (X) and target values (y)
X_train, y_train = [], []  # X_train will hold input sequences, y_train will hold corresponding target prices

# Create a sliding window for our stock (50 days)
for i in range(50, len(training_data)):  # Start from index 90 to ensure we have 90 previous days of data
    X_train.append(training_data[i-50:i, 0])  # Add sequence of 90 days (from i-90 to i-1) as input features
    y_train.append(training_data[i,0])  # Add the price at day i as the target value to predict
    
X_train, y_train = np.array(X_train), np.array(y_train)  # Convert Python lists to NumPy arrays for efficiency

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape to 3D: (samples, timesteps, features) for LSTM input


# # Microsoft Stock Price Prediction Analysis - Complete Documentation
# 
# ## Project Overview
# This notebook demonstrates end-to-end stock price prediction using Long Short-Term Memory (LSTM) neural networks. We analyze 35+ years of Microsoft stock data to build a model that can forecast future prices based on historical patterns.
# 
# ### Key Learning Objectives:
# 1. **Time Series Analysis**: Understanding stock market data patterns
# 2. **Data Preprocessing**: Scaling and sequence preparation for neural networks
# 3. **LSTM Architecture**: Building deep learning models for temporal data
# 4. **Model Evaluation**: Assessing prediction accuracy and visualization
# 
# ### Business Context:
# Stock price prediction is crucial for:
# - **Investment Decisions**: Helping investors make informed choices
# - **Risk Management**: Understanding potential price movements
# - **Portfolio Optimization**: Timing buy/sell decisions
# - **Market Analysis**: Identifying trends and patterns
# 
# ---
# 
# ## Step 1: Environment Setup and Library Imports
# 
# ### Purpose:
# Import all necessary libraries for data manipulation, visualization, and machine learning operations.
# 
# ### Libraries Explained:
# 

# In[32]:


# Build the Model - Create a sequential neural network architecture for time series prediction
model = keras.models.Sequential()  # Sequential model processes layers one after another in linear stack

# First Layer - LSTM layer with memory cells to capture long-term dependencies in stock price patterns
model.add(keras.layers.LSTM(64,  # 64 LSTM units (neurons) to learn complex temporal patterns
                           return_sequences=True,  # Return full sequence output to feed into next LSTM layer
                           input_shape=(X_train.shape[1],1)))  # Input shape: (60 timesteps, 1 feature)

# Second Layer - Another LSTM layer to learn higher-level temporal abstractions
model.add(keras.layers.LSTM(64,  # 64 LSTM units for deep feature extraction from sequences
                           return_sequences=False))  # Return only last output since no more LSTM layers follow

# 3rd Layer (Dense) - Fully connected layer to combine LSTM features for final prediction
model.add(keras.layers.Dense(128,  # 128 neurons to create rich feature combinations
                            activation="relu"))  # ReLU activation prevents vanishing gradients, adds non-linearity

# 4th Layer (Dropout) - Regularization technique to prevent overfitting by randomly ignoring neurons
model.add(keras.layers.Dropout(0.50))  # Randomly set 20% of neurons to zero during training to improve generalization

# Final Output Layer - Single neuron to output the predicted stock price
model.add(keras.layers.Dense(1))  # 1 output neuron for regression (predicting continuous price value), no activation for regression

# Configure Model Training Settings - Define how the model will learn from data
model.compile(optimizer="adam",  # Adam optimizer: adaptive learning rate algorithm for efficient gradient descent
              loss="mae",  # Mean Absolute Error: measures average absolute difference between predicted and actual prices
              metrics=[keras.metrics.RootMeanSquaredError()])  # RMSE metric: tracks square root of mean squared errors for evaluation

# Train the Model - Feed training data through network to learn price prediction patterns
training = model.fit(X_train, y_train,  # Input sequences (X_train) and target prices (y_train) for supervised learning
                    epochs=22,  # Number of complete passes through entire training dataset
                    batch_size=32)  # Process 32 samples at once before updating model weights (memory efficiency)

                    


# In[28]:


# Prep the test data
test_data = scaled_data[training_data_len - 90:]
X_test, y_test = [], dataset[training_data_len:]


for i in range(90, len(test_data)):
    X_test.append(test_data[i-90:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

# Make a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = df[:training_data_len]
test =  df[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(24,8))
plt.plot(train['Date'], train['Close'], label="Train (Actual)", color='blue')
plt.plot(test['Date'], test['Close'], label="Test (Actual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

