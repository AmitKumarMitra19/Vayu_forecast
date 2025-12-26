# Stock Forecasting Dashboard (LSTM vs N-BEATS)

A production-ready stock market forecasting application built using deep learning models (LSTM and N-BEATS) and deployed on Streamlit Cloud.  
The application supports single-stock analysis as well as side-by-side comparison of two stocks with next-day price prediction, direction inference, and RMSE-based evaluation.

---

## Project Overview

This project demonstrates an end-to-end machine learning pipeline for time series forecasting on stock market data using deep learning models.  
Live market data is fetched from Yahoo Finance, processed, and passed through trained models to generate predictions that are visualized in an interactive web dashboard.

---

## Key Features

- Deep learning-based stock price forecasting
- Single stock analysis with rolling forecast
- Side-by-side comparison of two stocks
- Next-day close price prediction
- Direction prediction (UP / DOWN)
- RMSE-based performance evaluation
- CPU-optimized inference
- Deployed on Streamlit Cloud

---

## Models Used

### LSTM (Long Short-Term Memory)
- Captures temporal dependencies in financial time series
- Suitable for sequential and noisy stock price data
- Used as the primary forecasting model

### N-BEATS
- Fully connected deep learning architecture for time series forecasting
- Strong baseline for univariate forecasting
- Used for comparative evaluation

---

## Side-by-Side Stock Comparison

The dashboard allows comparison of two stocks simultaneously based on:

- Last closing price
- Next-day predicted close
- Predicted direction (UP or DOWN)
- RMSE (prediction error)

Example output:
- AAPL: Direction = DOWN, RMSE = 0.3192
- MSFT: Direction = UP, RMSE = 0.1809

---

## System Architecture

Yahoo Finance (Live Data)  
↓  
Data Preprocessing and Scaling  
↓  
LSTM / N-BEATS Models  
↓  
Prediction and Evaluation (RMSE, Direction)  
↓  
Streamlit Cloud Dashboard

---

## Tech Stack

- Python
- PyTorch
- Streamlit
- scikit-learn
- yfinance
- NumPy
- Pandas

---

## Deployment

- Hosted on Streamlit Cloud
- CPU-only deployment (no GPU dependency)
- GitHub-based CI deployment
- Lightweight and production-safe

---

## Disclaimer

This project is intended strictly for educational and research purposes.  
It does not constitute financial advice, trading recommendations, or investment guidance.  
Stock market investments are subject to market risks.

---

## Recruiter Summary

Built and deployed a stock forecasting dashboard using LSTM and N-BEATS deep learning models with live market data, enabling side-by-side stock comparison, next-day price prediction, direction inference, and RMSE-based evaluation on Streamlit Cloud.

---

## Future Improvements

- Confidence intervals for predictions
- Risk-adjusted stock ranking
- Feature-based models using technical indicators
- Mobile-optimized UI

## Streamlit app: https://vayu-forecast.streamlit.app/

© 2025 Amit Kumar Mitra  
Built for educational and research purposes.


