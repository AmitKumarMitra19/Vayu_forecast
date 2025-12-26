import streamlit as st
import torch
import numpy as np
import yfinance as yf
import joblib
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# ---------------- CONFIG ----------------
LOOKBACK = 30
DEVICE = "cpu"

# ---------------- MODELS ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class NBeats(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LOOKBACK, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load("lstm_model.pth", map_location=DEVICE))
    lstm.eval()

    nbeats = NBeats()
    nbeats.load_state_dict(torch.load("nbeats_model.pth", map_location=DEVICE))
    nbeats.eval()

    return lstm, nbeats

lstm, nbeats = load_models()

# ---------------- UI ----------------
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("**Models:** LSTM vs N-BEATS (CPU optimized)")

ticker = st.text_input("Stock Ticker", "AAPL")
period = st.selectbox("Select Time Range", ["3mo", "6mo", "1y"])
model_choice = st.radio("Select Model", ["LSTM", "N-BEATS", "Compare Both"])

# ---------------- FETCH DATA ----------------
def fetch_data(ticker, period):
    data = yf.download(ticker, period=period)
    if data.empty or len(data) < LOOKBACK + 5:
        return None
    return data[["Close"]]

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    df = fetch_data(ticker, period)

    if df is None:
        st.error("Not enough data. Try a longer time range or a different stock.")
        st.stop()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i:i+LOOKBACK])
        y.append(scaled[i+LOOKBACK])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = np.array(y)

    # ---------------- LSTM ----------------
    lstm_preds = lstm(X).detach().numpy()
    lstm_rmse = math.sqrt(mean_squared_error(y, lstm_preds))
    lstm_price = scaler.inverse_transform([[lstm_preds[-1][0]]])[0][0]

    # ---------------- N-BEATS ----------------
    nbeats_preds = nbeats(X).detach().numpy()
    nbeats_rmse = math.sqrt(mean_squared_error(y, nbeats_preds))
    nbeats_price = scaler.inverse_transform([[nbeats_preds[-1][0]]])[0][0]

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Prediction Results")

    if model_choice in ["LSTM", "Compare Both"]:
        st.success(f"LSTM Next Close Prediction: ${lstm_price:.2f}")
        st.write(f"LSTM RMSE: {lstm_rmse:.4f}")

    if model_choice in ["N-BEATS", "Compare Both"]:
        st.success(f"N-BEATS Next Close Prediction: ${nbeats_price:.2f}")
        st.write(f"N-BEATS RMSE: {nbeats_rmse:.4f}")

    if model_choice == "Compare Both":
        st.subheader("📉 RMSE Comparison")
        st.bar_chart({
            "LSTM": lstm_rmse,
            "N-BEATS": nbeats_rmse
        })
