# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(page_title="Stock Analytics — ARMA - GARCH", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Stock Analytics Platform — ARMA - GARCH Forecast")
st.markdown("Enter any ticker (e.g., AAPL, MSFT, TSLA) — the app will load full historical data and produce charts, stats and ARMA+GARCH forecasts.")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Settings")
ticker_input = st.sidebar.text_input("Ticker symbol", "AAPL").upper().strip()
chart_type = st.sidebar.selectbox("Chart type", ["Candlestick", "Line"])
ma_short = st.sidebar.number_input("Short MA (period)", min_value=2, max_value=200, value=20)
ma_long = st.sidebar.number_input("Long MA (period)", min_value=2, max_value=500, value=50)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=14, value=5)
run_models = st.sidebar.button("Run Models & Forecast")

# -------------------------
# Helper: load ticker info & history
# -------------------------
@st.cache_data(ttl=3600)
def load_ticker_history(ticker):
    tk = yf.Ticker(ticker)
    # full history
    hist = tk.history(period="max", auto_adjust=False)  # keep raw OHLC for candlesticks
    info = tk.info if hasattr(tk, "info") else {}
    return hist, info

# -------------------------
# Load data
# -------------------------
if ticker_input == "":
    st.warning("Please enter a ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading data for {ticker_input} ..."):
    try:
        history, info = load_ticker_history(ticker_input)
    except Exception as e:
        st.error(f"Failed to load data for {ticker_input}: {e}")
        st.stop()

if history is None or history.empty:
    st.error(f"No historical data found for ticker: {ticker_input}. Check the symbol or try another ticker.")
    st.stop()

# Ensure columns exist and consistent names
# yfinance may return 'Close', 'Open', etc. Use them as-is for candlestick.
history = history.reset_index()
# Many yfinance returns use 'Date' column; ensure dtypes
if "Date" not in history.columns:
    history.rename(columns={history.columns[0]: "Date"}, inplace=True)
history["Date"] = pd.to_datetime(history["Date"])

# Some tickers return Adjusted Close as 'Adj Close' or 'Close' with auto_adjust; we'll compute returns from 'Adj Close' if present else Close
if "Adj Close" not in history.columns and "Close" in history.columns:
    history["Adj Close"] = history["Close"]

# Latest info
latest_row = history.iloc[-1]
latest_price = latest_row["Adj Close"]
latest_open = latest_row.get("Open", np.nan)
# Try to get current market price from ticker.info where available
market_price = None
market_change_pct = None
try:
    if info and isinstance(info, dict):
        market_price = info.get("regularMarketPrice", None)
        previous_close = info.get("previousClose", None)
        if market_price is not None and previous_close not in (None, 0):
            market_change_pct = (market_price - previous_close) / previous_close * 100
except Exception:
    market_price = None

# Header with company name and price metrics
col1, col2, col3, col4 = st.columns([3,2,2,2])
company_name = info.get("longName") if info and isinstance(info, dict) else None
col1.markdown(f"### {company_name or ticker_input}  —  **{ticker_input}**")
if market_price is not None:
    col2.metric("Current Market Price", f"{market_price:,.2f}")
else:
    col2.metric("Last Known Adj Close", f"{latest_price:,.2f}")
if market_change_pct is not None:
    col3.metric("Change vs Prev Close", f"{market_change_pct:.2f}%")
else:
    col3.metric("Latest Open", f"{latest_open:,.2f}")
col4.metric("Data Range", f"{history['Date'].dt.date.min()} ➜ {history['Date'].dt.date.max()}")

st.markdown("---")

# -------------------------
# Main layout: Chart + Stats
# -------------------------
left, right = st.columns((3,1))

with left:
    st.subheader("Price Chart — Full History")
    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=history["Date"],
                                     open=history["Open"],
                                     high=history["High"],
                                     low=history["Low"],
                                     close=history["Adj Close"],
                                     name="Price"))
    else:
        fig.add_trace(go.Scatter(x=history["Date"], y=history["Adj Close"], mode="lines", name="Adj Close"))
    # Add moving averages
    history['MA_short'] = history['Adj Close'].rolling(window=ma_short, min_periods=1).mean()
    history['MA_long'] = history['Adj Close'].rolling(window=ma_long, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=history["Date"], y=history["MA_short"], mode="lines", name=f"MA {ma_short}"))
    fig.add_trace(go.Scatter(x=history["Date"], y=history["MA_long"], mode="lines", name=f"MA {ma_long}"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=True, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume")
    vol_fig = px.bar(history, x="Date", y="Volume", labels={"Volume":"Volume"})
    st.plotly_chart(vol_fig, use_container_width=True)

with right:
    st.subheader("Quick Stats")
    st.write(f"**Ticker:** {ticker_input}")
    if company_name:
        st.write(f"**Company:** {company_name}")
    st.write(f"**Last adjusted close:** {latest_price:,.2f}")
    st.write(f"**Last open:** {latest_open:,.2f}")
    st.write(f"**Total data points:** {len(history):,}")
    # Basic summary metrics
    returns = np.log(history["Adj Close"] / history["Adj Close"].shift(1)).dropna()
    st.write("**Average daily return:**", f"{returns.mean():.6f}")
    st.write("**Daily return volatility (std):**", f"{returns.std():.6f}")
    st.write("**Max price:**", f"{history['Adj Close'].max():,.2f}")
    st.write("**Min price:**", f"{history['Adj Close'].min():,.2f}")

st.markdown("---")

# -------------------------
# ARMA + GARCH section
# -------------------------
st.header("ARMA + GARCH Modeling & Forecasting")

# Safety checks for enough data
min_required_points = 100  # conservative
if len(returns) < 30:
    st.warning("Not enough return data to build reliable ARMA/GARCH models. Need more historical points.")
else:
    # Allow user to run models explicitly
    if run_models:
        with st.spinner("Fitting ARMA (mean) model..."):
            try:
                # ARMA as ARIMA with d=0
                arma_order = (1, 0, 1)
                arma_model = ARIMA(returns, order=arma_order)
                arma_res = arma_model.fit()
                st.subheader("ARMA Model Summary")
                st.text(arma_res.summary().as_text())
            except Exception as e:
                st.error(f"ARMA model failed: {e}")
                arma_res = None

        if arma_res is not None:
            with st.spinner("Fitting GARCH(1,1) on residuals..."):
                try:
                    resid = arma_res.resid.dropna()
                    garch = arch_model(resid, vol="GARCH", p=1, q=1, dist="normal")
                    garch_res = garch.fit(disp="off")
                    st.subheader("GARCH Model Summary")
                    st.text(garch_res.summary().as_text())
                except Exception as e:
                    st.error(f"GARCH model failed: {e}")
                    garch_res = None

            # Forecasting
            if 'arma_res' in locals() and arma_res is not None and 'garch_res' in locals() and garch_res is not None:
                with st.spinner("Generating forecasts..."):
                    try:
                        # Forecast returns with ARMA for horizon (iterative)
                        arma_fore = arma_res.forecast(steps=forecast_horizon)
                        # arma_fore may be a Series or DataFrame; handle robustly
                        try:
                            # For statsmodels >=0.12 returns DataFrame with "mean"
                            if hasattr(arma_fore, 'predicted_mean'):
                                arma_mean = arma_fore.predicted_mean
                            else:
                                arma_mean = arma_fore.iloc[:, 0] if isinstance(arma_fore, pd.DataFrame) else arma_fore
                        except Exception:
                            arma_mean = arma_fore
                        # GARCH forecast of variance
                        garch_fore = garch_res.forecast(horizon=forecast_horizon, reindex=False)
                        var_fore = garch_fore.variance.values[-1]  # last row, array length = horizon
                        # Build forecast table
                        last_price_val = history["Adj Close"].iloc[-1]
                        forecast_rows = []
                        cumulative_price = last_price_val
                        for i in range(forecast_horizon):
                            mu = float(arma_mean.iloc[i]) if hasattr(arma_mean, "iloc") else float(arma_mean[i])
                            variance = float(var_fore[i])
                            sigma = np.sqrt(variance)
                            # price forecast using geometric Brownian-like step: S * exp(mu) — note: ignores stochastic term
                            forecast_price = cumulative_price * np.exp(mu)
                            forecast_rows.append({
                                "Day": i+1,
                                "Forecast_Return": mu,
                                "Forecast_Variance": variance,
                                "Forecast_Volatility(%)": sigma*100,
                                "Forecasted_Price": forecast_price
                            })
                            # update cumulative price for iterative forecast (use mu only)
                            cumulative_price = forecast_price

                        forecast_df = pd.DataFrame(forecast_rows)
                        st.subheader("Forecast Table")
                        st.dataframe(forecast_df.style.format({
                            "Forecast_Return": "{:.6f}",
                            "Forecast_Variance": "{:.8f}",
                            "Forecast_Volatility(%)": "{:.4f}",
                            "Forecasted_Price": "{:,.2f}"
                        }))

                        # Plot forecasted price vs historical tail
                        plot_len = min(200, len(history))
                        tail_hist = history.tail(plot_len).copy()
                        # construct forecast timeline
                        last_date = history["Date"].iloc[-1]
                        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_horizon)]
                        forecast_plot = go.Figure()
                        forecast_plot.add_trace(go.Scatter(x=tail_hist["Date"], y=tail_hist["Adj Close"], mode="lines", name="History"))
                        forecast_plot.add_trace(go.Scatter(x=forecast_dates, y=forecast_df["Forecasted_Price"], mode="lines+markers", name="Forecast"))
                        forecast_plot.update_layout(title="Historical (tail) + Forecasted Prices", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(forecast_plot, use_container_width=True)

                        # Download button for forecast
                        csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Forecast CSV", data=csv_bytes, file_name=f"{ticker_input}_forecast.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Forecast generation failed: {e}")
    else:
        st.info("Click 'Run Models & Forecast' in the sidebar to fit ARMA+GARCH and produce forecasts.")

st.markdown("---")
st.markdown("")
st.markdown("Tip: Try other tickers like MSFT, TSLA, GOOGL. Models may require enough history to fit reliably.")
