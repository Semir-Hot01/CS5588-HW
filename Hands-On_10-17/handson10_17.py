import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model
import networkx as nx

info = ['sector', 'industry', 'fullTimeEmployees', 'profitMargins',
        'operatingMargins', 'returnOnAssets', 'returnOnEquity',
        'revenueGrowth', 'earningsGrowth', 'debtToEquity', 'totalCash',
        'totalDebt', 'totalRevenue', 'bookValue', 'operatingCashflow',
        'freeCashflow', 'targetLowPrice', 'targetMeanPrice',
        'targetMedianPrice', 'recommendationMean']

data = []
stocks = ["AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "TSLA", "NVDA", "META", "UNH", "JNJ", "V", "JPM", "PG", "HD", "MA", "BAC", "XOM", "CVX", "LLY", "PFE"]

# Download closing prices
prices = yf.download(stocks, period="1d")['Close']

for stock in stocks:
  ticker = yf.Ticker(stock)
  stock_info = [stock]
  for i in info:
    try:
      stock_info.append(ticker.info[i])
    except KeyError:
      stock_info.append(None)
  # Add closing price to stock_info
  stock_info.append(prices[stock].iloc[0])
  data.append(stock_info)

df = pd.DataFrame(data, columns = ['Ticker'] + info + ['Close'])

categorical_features = ['Ticker', 'industry', 'sector']  # Include 'Ticker', 'industry', and 'sector' as categorical
numerical_features = df.drop(['Ticker', 'Close', 'industry', 'sector'], axis=1).columns.tolist() # Exclude categorical features

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')) # Handle unknown categories during testing
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Features and target
X = df.drop(['Close'], axis=1)  # Drop 'Ticker' column along with 'Close'
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Neural Network"""

# Feature scaling
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Build a deep neural network model for stock prediction
nn_model = Sequential([
  Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
  Dense(64, activation='relu'),
  Dense(1) # Output layer for regression
])

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_nn = nn_model.predict(X_test_scaled)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f"Neural Network MSE: {mse_nn}")

"""Ensemble Model (RandomForest + Neural Network)"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)
nn_preds = nn_model.predict(X_test_scaled)

ensemble_preds = np.mean([rf_preds, nn_preds.flatten()], axis=0)
mse_ensemble = mean_squared_error(y_test, ensemble_preds)
print(f"Ensemble Model MSE: {mse_ensemble}")

"""Stock Price Forecasting Using Regression (LSTM Model)"""

df = df[['Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

def create_dataset(data, time_step=50):
  X, y = [], []
  for i in range(time_step, len(data)):
    X.append(data[i-time_step:i, 0])
    y.append(data[i, 0])
  return np.array(X), np.array(y)

time_step = 1
X, y = create_dataset(scaled_data, time_step)

X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lstm_model = Sequential([
  LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
  LSTM(50, return_sequences=False),
  Dense(25),
  Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mse_lstm = mean_squared_error(y_test_rescaled, y_pred_lstm_rescaled)
print(f"LSTM Model MSE: {mse_lstm}")

"""Building a Flask Application for Stock Price Prediction, Forecasting, and Knowledge Graph

"""

rf_model = joblib.load('rf_stock_model.pkl')
nn_model = load_model('nn_stock_model.h5')
lstm_model = load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  data = request.json
  features = np.array([data['earningsGrowth'], data['debtToEquity'], data['totalCash'],
              data['totalDebt'], data['totalRevenue'], data['bookValue']]).reshape(1, -1)

  features_scaled = scaler.transform(features)
  rf_pred = rf_model.predict(features_scaled)
  nn_pred = nn_model.predict(features_scaled)
  final_pred = np.mean([rf_pred, nn_pred.flatten()])
  recommendation = get_recommendation(final_pred)
  return jsonify({
  'predicted_stock_price': final_pred,
  'recommendation': recommendation
  })

@app.route('/forecast', methods=['POST'])
def forecast():
  data = request.json
  historical_data = np.array(data['historical_prices']).reshape(-1, 1)
  historical_data_scaled = scaler.transform(historical_data)
  time_step = 50
  X = historical_data_scaled[-time_step:].reshape((1, time_step, 1))
  forecasted_price_scaled = lstm_model.predict(X)
  forecasted_price = scaler.inverse_transform(forecasted_price_scaled)
  return jsonify({
  'forecasted_stock_price': forecasted_price[0][0]
  })
