import streamlit as st
import torch
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# ---------------- Models ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class NBeats(nn.Module):
    def __init__(self, lookback=30):
        super().__init__()
        self.fc1 = nn.Linear(lookback, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- Load Models ----------------
LOOKBACK = 30

lstm = LSTMModel()
lstm.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
lstm.eval()

nbeats = NBeats()
nbeats.load_state_dict(torch.load("nbeats_model.pth", map_location="cpu"))
nbeats.eval()

scaler = joblib.load("scaler.save")

# ---------------- Streamlit UI ----------------
st.title("📈 Stock Price Prediction (LSTM vs N-BEATS)")
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Predict"):
    data = yf.download(ticker, period="3mo")[["Close"]]
    scaled = scaler.transform(data)

    seq = scaled[-LOOKBACK:]
    seq = torch.tensor(seq).float().unsqueeze(0)

    lstm_pred = lstm(seq).item()
    nbeats_pred = nbeats(seq).item()

    lstm_price = scaler.inverse_transform([[lstm_pred]])[0][0]
    nbeats_price = scaler.inverse_transform([[nbeats_pred]])[0][0]

    st.success(f"LSTM Prediction: ${lstm_price:.2f}")
    st.success(f"N-BEATS Prediction: ${nbeats_price:.2f}")
