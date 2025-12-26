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

def prepare_sequences(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i:i+LOOKBACK])
        y.append(float(scaled[i+LOOKBACK][0]))

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        np.array(y, dtype=float),
        scaler,
        scaled
    )

def rolling_forecast(model, raw_data, scaler, steps):
    preds = []
    temp = raw_data.copy()

    for _ in range(steps):
        seq = temp[-LOOKBACK:]
        seq_scaled = scaler.transform(seq)
        X = torch.tensor(seq_scaled).float().unsqueeze(0)
        pred = float(model(X).item())
        preds.append(pred)
        temp = np.vstack([temp, [[pred]]])

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def walk_forward_rmse(model, scaled_data):
    rmses = []

    for i in range(LOOKBACK, len(scaled_data)):
        X = torch.tensor(
            scaled_data[i-LOOKBACK:i].reshape(1, LOOKBACK, 1),
            dtype=torch.float32
        )
        y_true = float(scaled_data[i][0])
        y_pred = float(model(X).item())
        rmses.append(abs(y_true - y_pred))

    return rmses

# ---------------- UI ----------------
st.title("📈 Vayu Stock Forecasting Dashboard")
st.markdown("**LSTM & N-BEATS | CPU-Optimized | Streamlit Cloud Ready**")

st.subheader("🔹 Single Stock Analysis")
ticker = st.text_input("Stock Ticker", "AAPL")
period = st.selectbox("Time Range", ["3mo", "6mo", "1y"])
model_choice = st.radio("Model Selection", ["LSTM", "N-BEATS", "Compare Both"])
forecast_days = st.slider("Rolling Forecast Days", 5, 30, 10)

# ---------------- SINGLE STOCK RUN ----------------
if st.button("Run Single Stock Prediction"):
    df = fetch_data(ticker, period)

    if df is None:
        st.error("Not enough data. Try a longer time range or another stock.")
        st.stop()

    X, y, scaler, scaled = prepare_sequences(df.values)

    lstm_preds = lstm(X).detach().numpy().flatten()
    nbeats_preds = nbeats(X).detach().numpy().flatten()

    lstm_rmse = math.sqrt(mean_squared_error(y, lstm_preds))
    nbeats_rmse = math.sqrt(mean_squared_error(y, nbeats_preds))

    lstm_price = float(scaler.inverse_transform([[lstm_preds[-1]]])[0][0])
    nbeats_price = float(scaler.inverse_transform([[nbeats_preds[-1]]])[0][0])

    last_close = float(df["Close"].iloc[-1])

    st.subheader("📊 Prediction Results")

    if model_choice in ["LSTM", "Compare Both"]:
        st.success(f"LSTM Next Close: ${lstm_price:.2f}")
        st.write(f"LSTM RMSE: {lstm_rmse:.4f}")
        st.write("Direction:", "UP 📈" if lstm_price > last_close else "DOWN 📉")

    if model_choice in ["N-BEATS", "Compare Both"]:
        st.success(f"N-BEATS Next Close: ${nbeats_price:.2f}")
        st.write(f"N-BEATS RMSE: {nbeats_rmse:.4f}")
        st.write("Direction:", "UP 📈" if nbeats_price > last_close else "DOWN 📉")

    if model_choice == "Compare Both":
        st.subheader("📉 RMSE Comparison")
        st.bar_chart({"LSTM": lstm_rmse, "N-BEATS": nbeats_rmse})

    st.subheader("📈 Actual vs Predicted (LSTM)")
    overlay_df = pd.DataFrame({
        "Actual": scaler.inverse_transform(y.reshape(-1, 1)).flatten()[-100:],
        "Predicted": scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()[-100:]
    })
    st.line_chart(overlay_df)

    st.subheader("🔄 Rolling Forecast")
    rolling_preds = rolling_forecast(lstm, df.values, scaler, forecast_days)
    st.line_chart(rolling_preds)

    st.subheader("📊 Walk-Forward RMSE Trend")
    rmse_trend = walk_forward_rmse(lstm, scaled)
    st.line_chart(rmse_trend)

# ---------------- MULTI-STOCK COMPARISON ----------------
st.markdown("---")
st.subheader("📊 Multi-Stock Comparison (LSTM)")

tickers_input = st.text_input(
    "Enter multiple tickers (comma-separated)",
    "AAPL,MSFT,GOOGL"
)

if st.button("Compare Stocks"):
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    results = []

    for t in tickers:
        df = fetch_data(t, period)
        if df is None:
            continue

        X, y, scaler, _ = prepare_sequences(df.values)
        preds = lstm(X).detach().numpy().flatten()

        rmse = math.sqrt(mean_squared_error(y, preds))
        next_price = float(scaler.inverse_transform([[preds[-1]]])[0][0])
        last_close = float(df["Close"].iloc[-1])

        results.append({
            "Stock": t,
            "Last Close": round(last_close, 2),
            "Predicted Close": round(next_price, 2),
            "Direction": "UP 📈" if next_price > last_close else "DOWN 📉",
            "RMSE": round(rmse, 4)
        })

    if results:
        result_df = pd.DataFrame(results)
        st.dataframe(result_df)
        st.bar_chart(result_df.set_index("Stock")["RMSE"])
    else:
        st.warning("No valid data found for the selected stocks.")

# ---------------- DISCLAIMER ----------------
st.markdown("""
---
⚠️ **Disclaimer**  
This application is for **educational and research purposes only**.  
It does **not constitute financial advice, investment recommendations, or trading signals**.  
Stock market investments are subject to market risks.
""")
