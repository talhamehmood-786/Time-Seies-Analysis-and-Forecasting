import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ---------------------------------------------------
# 📊 Streamlit App Config
# ---------------------------------------------------
st.set_page_config(page_title="📈 ARMA + GARCH Stock Forecast", layout="wide")
st.title("📊 Apple Stock Forecast (ARMA + GARCH Model)")
st.markdown("### 🔍 A professional stock volatility forecasting dashboard")

# ---------------------------------------------------
# 📌 Basic Info
# ---------------------------------------------------
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# ---------------------------------------------------
# 📥 Step 1: Get the Data
# ---------------------------------------------------
st.subheader("📥 Step 1: Download Historical Data")

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if data.empty:
    st.error("❌ No data found. Please check the ticker symbol or date range.")
    st.stop()

if "Close" in data.columns and "Adj Close" not in data.columns:
    data.rename(columns={"Close": "Adj Close"}, inplace=True)

st.dataframe(data.tail())

# ---------------------------------------------------
# 📈 Step 2: Calculate Log Returns
# ---------------------------------------------------
st.subheader("📈 Step 2: Calculate Returns")

data["Returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
data.dropna(inplace=True)
st.line_chart(data["Returns"], height=300)

# ---------------------------------------------------
# 🤖 Step 3: Fit ARMA Model
# ---------------------------------------------------
st.subheader("🤖 Step 3: ARMA Model on Returns")

model_arma = ARIMA(data["Returns"], order=(1, 0, 1))
arma_result = model_arma.fit()
st.code(arma_result.summary().as_text())

# ---------------------------------------------------
# 📉 Step 4: Fit GARCH on Residuals
# ---------------------------------------------------
st.subheader("📉 Step 4: GARCH Volatility Model")

residuals = arma_result.resid
model_garch = arch_model(residuals, vol="GARCH", p=1, q=1)
garch_result = model_garch.fit(disp="off")
st.code(garch_result.summary().as_text())

# ---------------------------------------------------
# 🔮 Step 5: Forecast Returns & Volatility
# ---------------------------------------------------
st.subheader("🔮 Step 5: Forecast Returns and Volatility")

arma_forecast = arma_result.forecast(steps=1)
predicted_return = arma_forecast.iloc[0]

st.success(f"📈 Forecasted Return (Next Day): {predicted_return:.6f} ({predicted_return*100:.2f}%)")

garch_forecast = garch_result.forecast(horizon=5)
variance_forecast = garch_forecast.variance.iloc[-1]

st.markdown("### 📊 5-Day Volatility Forecast")
vol_df = pd.DataFrame({
    "Day": [f"Day {i+1}" for i in range(5)],
    "Variance": variance_forecast.values,
    "Volatility (%)": np.sqrt(variance_forecast.values) * 100
})
st.dataframe(vol_df)

# ---------------------------------------------------
# 📌 Step 6: Forecasted Price
# ---------------------------------------------------
st.subheader("📌 Step 6: Forecasted Price")

last_price = data["Adj Close"].iloc[-1]
forecasted_price = last_price * np.exp(predicted_return)

st.metric(label="📌 Last Known Price", value=f"${last_price:.2f}")
st.metric(label="📈 Forecasted Price (Next Day)", value=f"${forecasted_price:.2f}")

# ---------------------------------------------------
# 📅 Step 7: Actual Data Comparison
# ---------------------------------------------------
st.subheader("📅 Step 7: Actual Price Comparison")

future_data = yf.download(ticker, start="2024-01-02", end="2024-01-04", auto_adjust=True)
if not future_data.empty:
    if "Close" in future_data.columns and "Adj Close" not in future_data.columns:
        future_data.rename(columns={"Close": "Adj Close"}, inplace=True)

    st.dataframe(future_data)
    actual_price = future_data["Adj Close"].iloc[0]
    error_pct = ((forecasted_price - actual_price) / actual_price) * 100

    st.info(f"📊 Actual Price on 2024-01-02: ${actual_price:.2f}")
    st.warning(f"📉 Forecast Error: {error_pct:.2f}%")
else:
    st.warning("⚠️ Actual price data not available yet for comparison.")

# ---------------------------------------------------
# 📌 Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown("✅ Built with **Streamlit**, **yfinance**, **statsmodels**, and **arch** for time series forecasting.")
st.markdown("💡 Tip: Replace 'AAPL' with 'MSFT', 'TSLA', 'GOOGL', etc. to analyze other stocks.")
