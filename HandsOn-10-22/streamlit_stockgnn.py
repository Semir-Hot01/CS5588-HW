import yfinance as yf
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st

!pip install streamlit
!pip install torch
!pip install torch torch-geometric pandas scikit-learn

"""# Inital pull of data to see what it looks like"""

stocks = ["AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "TSLA", "NVDA", "META", "UNH", "JNJ",
          "V", "JPM", "PG", "HD", "MA", "BAC", "XOM", "CVX", "LLY", "PFE"]

data = yf.download(stocks, period="1mo")
df = pd.DataFrame(data)
df.to_csv("stock_data.csv")

df.head()

"""# Fine tuned data API call

just pulling the stock data would only give us the prices at the end of that, that is not what we would be looking at for our project, we would want some more data that we could base our model off.
"""

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

df.head()

df['Close']

df.to_csv("stock_data.csv")

"""# Modeling"""

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd

print(df.isna().sum())

df_cleaned = df.dropna()

df_cleaned = df.dropna(how='any').reset_index(drop=True)

df_cleaned.isna().sum()

from sklearn.preprocessing import StandardScaler


df_encoded = pd.get_dummies(df_cleaned.drop(columns=['Ticker']), columns=['sector', 'industry'], drop_first=True)
df_encoded = df_encoded.astype(int)

features = ['fullTimeEmployees', 'profitMargins', 'operatingMargins', 'returnOnAssets',
            'returnOnEquity', 'revenueGrowth', 'earningsGrowth', 'debtToEquity',
            'totalCash', 'totalDebt', 'totalRevenue', 'bookValue',
            'operatingCashflow', 'freeCashflow', 'targetLowPrice',
            'targetMeanPrice', 'targetMedianPrice', 'recommendationMean']

scaler = StandardScaler()
df_encoded[features] = scaler.fit_transform(df_encoded[features])

df_encoded.head()

X = df_encoded.drop(columns=['Close']).values
y = df_encoded['Close'].values

# Convert to torch tensors
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

edge_index = []

for i in range(len(df_cleaned)):
    for j in range(i + 1, len(df_cleaned)):
        if df_cleaned.iloc[i]['sector'] == df.iloc[j]['sector']:
            edge_index.append([i, j])
            edge_index.append([j, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index, y=y)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels, activation):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.fc = nn.Linear(hidden_channels_2, out_channels)
        self.activation = activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.fc(x)
        return x

param_grid = {
    'in_channels': [x.shape[1]],
    'hidden_channels_1': [32, 64, 128],
    'hidden_channels_2': [16, 32, 64],
    'out_channels': [1],
    'activation': [torch.relu, torch.sigmoid],
    'lr': [0.001, 0.01, 0.1],
    'epochs': [100, 200, 300]
}

from sklearn.model_selection import GridSearchCV

import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin

class PyTorchEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model, epochs=100, lr=0.001, criterion=nn.MSELoss(), optimizer=optim.Adam):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, y=None):
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        edge_index = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if i % 2 == j % 2:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=X, edge_index=edge_index, y=y)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.flatten(), data.y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            data = Data(x=X, edge_index=edge_index)

            output = self.model(data).flatten().detach().numpy()
        return output

    def score(self, X, y=None):
        predictions = self.predict(X)
        return -1 * np.mean((predictions - y)**2)

model = GNN(in_channels=x.shape[1], hidden_channels_1=64, hidden_channels_2=32, out_channels=1, activation=torch.relu)
estimator = PyTorchEstimator(model=model)

param_grid = {
    'epochs': [100, 200],
    'lr': [0.001, 0.01]
}

grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=2)
grid_search.fit(X, y)

print("Best parameters found: ", grid_search.best_params_)

best_params = grid_search.best_params_

model = GNN(in_channels=x.shape[1],
            hidden_channels_1=best_params['hidden_channels_1'],
            hidden_channels_2=best_params['hidden_channels_2'],
            out_channels=1,
            activation=best_params['activation'])

optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

model.train()
for epoch in range(best_params['epochs']):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output.flatten(), data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

import torch.optim as optim


optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

model.train()
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output.flatten(), data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    predictions = model(data).flatten().detach().numpy()

actual_prices = y.detach().numpy()
tickers = df_cleaned['Ticker'].values

comparison_df = pd.DataFrame({'Ticker': tickers, 'Actual Price': actual_prices, 'Predicted Price': predictions})
print(comparison_df)

results_df = pd.DataFrame({'Ticker': df_cleaned['Ticker'], 'Predicted Price': predictions})
print(results_df)

"""# Streamlit Portion"""

def predict_stock_price(stock_ticker):
    ticker = yf.Ticker(stock_ticker)
    with torch.no_grad():
        prediction = model(data).item()

    return prediction

st.title('Stock Price Prediction')

stock_ticker = st.text_input('Enter stock ticker:', 'AAPL')
if st.button('Predict'):
    predicted_price = predict_stock_price(stock_ticker)
    st.write(f'Predicted price for {stock_ticker}: {predicted_price}')


    st.line_chart(pd.DataFrame({'Price': [predicted_price]}))
