from datetime import datetime
import streamlit as st
import torch
import numpy as np
import yfinance as yf
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
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

# ---------------- FUNCTIONS ----------------
def fetch_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    if data.empty or len(data) < LOOKBACK + 5:
        return None
    return data[["Close"]]

def run_model(model, df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i:i+LOOKBACK])
        y.append(float(scaled[i+LOOKBACK][0]))

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = np.array(y)

    preds = model(X).detach().numpy().flatten()
    rmse = math.sqrt(mean_squared_error(y, preds))
    next_price = float(scaler.inverse_transform([[preds[-1]]])[0][0])
    last_close = float(df.iloc[-1][0])

    return next_price, rmse, last_close

# ---------------- UI ----------------
st.title("📈 Stock Forecasting Dashboard")
st.markdown("**LSTM vs N-BEATS | CPU | Streamlit Cloud Ready**")

st.subheader("📊 Side-by-Side Stock Comparison (Up to 2 Stocks)")

col1, col2 = st.columns(2)
with col1:
    ticker_1 = st.text_input("Stock A Ticker", "AAPL")
with col2:
    ticker_2 = st.text_input("Stock B Ticker", "MSFT")

period = st.selectbox("Time Range", ["3mo", "6mo", "1y"])
model_choice = st.radio("Model Selection", ["LSTM", "N-BEATS"])

if st.button("Run Side-by-Side Comparison"):
    model = lstm if model_choice == "LSTM" else nbeats

    df1 = fetch_data(ticker_1, period)
    df2 = fetch_data(ticker_2, period)

    if df1 is None or df2 is None:
        st.error("One or both stocks do not have enough data.")
        st.stop()

    p1, rmse1, last1 = run_model(model, df1)
    p2, rmse2, last2 = run_model(model, df2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📌 {ticker_1}")
        st.metric("Last Close", f"${last1:.2f}")
        st.metric("Predicted Close", f"${p1:.2f}")
        st.metric("Direction", "UP 📈" if p1 > last1 else "DOWN 📉")
        st.metric("RMSE", f"{rmse1:.4f}")

    with col2:
        st.subheader(f"📌 {ticker_2}")
        st.metric("Last Close", f"${last2:.2f}")
        st.metric("Predicted Close", f"${p2:.2f}")
        st.metric("Direction", "UP 📈" if p2 > last2 else "DOWN 📉")
        st.metric("RMSE", f"{rmse2:.4f}")

# ---------------- FIXED FOOTER ----------------
current_year = datetime.now().year

footer = f"""
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f9f9f9;
    color: #6c757d;
    text-align: center;
    font-size: 12px;
    padding: 10px;
    border-top: 1px solid #e0e0e0;
    z-index: 100;
}}
</style>

<div class="footer">
© {current_year} Amit Kumar Mitra. All rights reserved.<br>
This application is for educational and research purposes only and does not constitute financial advice or investment recommendations.
Stock market investments are subject to market risks.
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

