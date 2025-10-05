# Time-Seies-Analysis-and-Forecasting
Stock Analytics Dashboard — ARMA + GARCH Forecasting

This project is a professional stock analytics platform built with Streamlit, yfinance, Plotly, and advanced time series models (ARMA and GARCH) to analyze, visualize, and forecast stock prices and volatility.
It automatically fetches full historical data for any stock ticker (e.g., AAPL, MSFT, TSLA) without any date restriction and provides forecasting, risk analysis, and interactive visualization — all in one app.

🚀 Features

📊 Full Historical Data – Loads the complete available history for any stock ticker (no date selection needed).

🏢 Company Overview in Header – Shows company name, latest price, % change, and opening price.

🕯️ Interactive Charts – Choose between Candlestick and Line charts with short & long moving averages.

📈 Price Metrics & Stats – Key statistics like average return, volatility, max/min price, and more.

🤖 ARMA (Mean) + GARCH (Volatility) Models – Forecast future returns and volatility with advanced time series analysis.

🔮 Multi-Day Price Forecast – Predict future stock prices for the next N days (configurable by the user).

📉 Forecast Table & Download – Detailed forecast table with returns, variance, volatility %, and predicted price.

📊 Forecast vs Historical Chart – Compare predicted future prices with recent historical data.

📤 Downloadable Forecast CSV – Export forecast results with one click.

📁 Project Structure
stock-analytics-dashboard/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

🛠️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/stock-analytics-dashboard.git
cd stock-analytics-dashboard


Install the required packages

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Then open your browser at 👉 http://localhost:8501

☁️ Deployment (Streamlit Cloud)

Push your code to GitHub

Go to https://share.streamlit.io

Click New app → Select your repo → app.py

Click Deploy 🚀

Your live app will be available at:

https://your-username-stock-dashboard.streamlit.app

📊 Supported Stocks

Enter any valid ticker symbol, such as:

AAPL – Apple Inc.

MSFT – Microsoft Corporation

GOOGL – Alphabet Inc.

TSLA – Tesla Inc.

AMZN – Amazon.com Inc.

The app will fetch full available historical data directly from Yahoo Finance.

🧠 Models Used

ARMA (AutoRegressive Moving Average): Models the mean behavior of log returns.

GARCH (Generalized AutoRegressive Conditional Heteroskedasticity): Models and forecasts volatility (variance) in returns.

These models combined allow us to forecast not just price direction, but also expected volatility and risk levels over future days.

📈 Example Use Cases

📊 Quantitative analysis of stock returns and volatility

🔮 Forecasting short-term stock prices and risk

📉 Historical trend visualization for investment decisions

🧠 Research and academic projects on financial time series

🤝 Contributing

Contributions are welcome! Feel free to fork the repo, create a new branch, and submit a pull request.

📜 License

This project is licensed under the MIT License — feel free to use and modify it for personal or commercial projects.
