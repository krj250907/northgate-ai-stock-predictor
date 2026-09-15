import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

sys.path.append(
    os.path.join(ROOT_DIR, "src")
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Northgate AI Stock Predictor",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("Northgate AI Stock Predictor")
st.caption(
    "AI-powered stock forecasting, sentiment analysis, "
    "risk analytics and portfolio recommendations"
)

st.warning(
    "Educational use only. This dashboard is not financial advice. "
    "Predictions are model outputs and can be wrong."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_forecasts():
    path = os.path.join(
        REPORTS_DIR,
        "model_forecasts.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_recommendations():
    path = os.path.join(
        REPORTS_DIR,
        "stock_recommendations.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_prices(ticker):
    path = os.path.join(
        PROCESSED_DIR,
        f"{ticker}.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    return df


forecasts = load_forecasts()
recommendations = load_recommendations()


# ============================================================
# CHECK DATA
# ============================================================

if forecasts.empty:
    st.error(
        "Forecast file not found. "
        "Run src/generate_forecasts.py first."
    )
    st.stop()

if recommendations.empty:
    st.error(
        "Recommendation file not found. "
        "Run src/recommend_pipeline.py first."
    )
    st.stop()


# ============================================================
# CLEAN RECOMMENDATIONS
# ============================================================

if "Ticker" not in recommendations.columns:
    recommendations = recommendations.reset_index()

recommendations["Predicted_Return"] = pd.to_numeric(
    recommendations["Predicted_Return"],
    errors="coerce",
)

recommendations["Composite_Score"] = pd.to_numeric(
    recommendations["Composite_Score"],
    errors="coerce",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

available_tickers = sorted(
    recommendations["Ticker"].dropna().unique()
)

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    available_tickers,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.header("Market Overview")

buy_count = (
    recommendations["Recommendation"]
    .eq("BUY")
    .sum()
)

hold_count = (
    recommendations["Recommendation"]
    .eq("HOLD")
    .sum()
)

sell_count = (
    recommendations["Recommendation"]
    .eq("SELL")
    .sum()
)

latest_date = forecasts["Latest_Date"].max()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "BUY",
    int(buy_count),
)

col2.metric(
    "HOLD",
    int(hold_count),
)

col3.metric(
    "SELL",
    int(sell_count),
)

col4.metric(
    "Forecast Date",
    str(latest_date),
)


# ============================================================
# RECOMMENDATION TABLE
# ============================================================

st.subheader("Stock Recommendations")

display_df = recommendations[
    [
        "Ticker",
        "Predicted_Return",
        "Sentiment",
        "Volatility",
        "Beta",
        "Composite_Score",
        "Recommendation",
    ]
].copy()

display_df["Predicted_Return"] = (
    display_df["Predicted_Return"] * 100
)

display_df["Predicted_Return"] = (
    display_df["Predicted_Return"].round(2)
)

display_df["Sentiment"] = (
    display_df["Sentiment"].round(3)
)

display_df["Volatility"] = (
    display_df["Volatility"] * 100
).round(2)

display_df["Beta"] = (
    display_df["Beta"].round(2)
)

display_df["Composite_Score"] = (
    display_df["Composite_Score"].round(3)
)

display_df = display_df.rename(
    columns={
        "Predicted_Return": "Forecast (%)",
        "Sentiment": "Sentiment",
        "Volatility": "Volatility (%)",
        "Beta": "Beta",
        "Composite_Score": "Composite Score",
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SELECTED STOCK
# ============================================================

st.header(f"{selected_ticker} Analysis")

selected = recommendations[
    recommendations["Ticker"] == selected_ticker
]

if selected.empty:
    st.warning(
        "No recommendation data available."
    )
    st.stop()

row = selected.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Recommendation",
    row["Recommendation"],
)

col2.metric(
    "Predicted Return",
    f"{row['Predicted_Return'] * 100:.2f}%",
)

col3.metric(
    "Composite Score",
    f"{row['Composite_Score']:.3f}",
)

col4.metric(
    "Beta",
    f"{row['Beta']:.2f}",
)


# ============================================================
# PRICE CHART
# ============================================================

prices = load_prices(selected_ticker)

if not prices.empty:

    st.subheader("Historical Price")

    fig_price = px.line(
        prices,
        x="Date",
        y="Close",
        title=f"{selected_ticker} Closing Price",
    )

    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True,
    )


# ============================================================
# SIGNAL BREAKDOWN
# ============================================================

st.subheader("Signal Breakdown")

signal_df = pd.DataFrame(
    {
        "Signal": [
            "Forecast",
            "Sentiment",
            "Volatility",
            "Beta",
        ],
        "Value": [
            row["Predicted_Return"],
            row["Sentiment"],
            -row["Volatility"],
            -row["Beta"],
        ],
    }
)

fig_signal = px.bar(
    signal_df,
    x="Signal",
    y="Value",
    title="Model Signal Components",
)

fig_signal.update_layout(
    xaxis_title="",
    yaxis_title="Signal Value",
)

st.plotly_chart(
    fig_signal,
    use_container_width=True,
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Northgate AI Stock Predictor uses historical market data, "
    "machine-learning forecasts, sentiment signals and risk metrics. "
    "Past performance does not guarantee future results."
)