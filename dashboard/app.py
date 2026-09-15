import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORTS_DIR = os.path.join(
    ROOT_DIR,
    "reports"
)

PROCESSED_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed"
)

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

st.title(
    "Northgate AI Stock Predictor"
)

st.caption(
    "AI-powered stock forecasting, sentiment analysis, "
    "risk analytics and portfolio recommendations"
)

st.warning(
    "Educational use only. This dashboard is not financial advice. "
    "Predictions are model outputs and can be wrong."
)


# ============================================================
# LOAD FORECASTS
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


# ============================================================
# LOAD RECOMMENDATIONS
# ============================================================

@st.cache_data
def load_recommendations():

    path = os.path.join(
        REPORTS_DIR,
        "stock_recommendations.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# LOAD STOCK PRICES
# ============================================================

@st.cache_data
def load_prices(ticker):

    path = os.path.join(
        PROCESSED_DIR,
        f"{ticker}.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(
        path,
        parse_dates=["Date"],
    )


# ============================================================
# LOAD DATA
# ============================================================

forecasts = load_forecasts()

recommendations = load_recommendations()


# ============================================================
# CHECK REQUIRED DATA
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

    recommendations = (
        recommendations
        .reset_index()
    )


numeric_columns = [
    "Predicted_Return",
    "Sentiment",
    "Volatility",
    "Beta",
    "Composite_Score",
]

for column in numeric_columns:

    if column in recommendations.columns:

        recommendations[column] = pd.to_numeric(
            recommendations[column],
            errors="coerce",
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Controls"
)

available_tickers = sorted(
    recommendations["Ticker"]
    .dropna()
    .unique()
)

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    available_tickers,
)


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.header(
    "Market Overview"
)

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

latest_date = forecasts[
    "Latest_Date"
].max()


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
# STOCK RECOMMENDATIONS
# ============================================================

st.subheader(
    "Stock Recommendations"
)

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
).round(2)

display_df["Sentiment"] = (
    display_df["Sentiment"]
    .round(3)
)

display_df["Volatility"] = (
    display_df["Volatility"] * 100
).round(2)

display_df["Beta"] = (
    display_df["Beta"]
    .round(2)
)

display_df["Composite_Score"] = (
    display_df["Composite_Score"]
    .round(3)
)


display_df = display_df.rename(
    columns={
        "Predicted_Return": "Forecast (%)",
        "Volatility": "Volatility (%)",
        "Composite_Score": "Composite Score",
    }
)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# SELECTED STOCK ANALYSIS
# ============================================================

st.header(
    f"{selected_ticker} Analysis"
)

selected = recommendations[
    recommendations["Ticker"]
    == selected_ticker
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
# HISTORICAL PRICE
# ============================================================

prices = load_prices(
    selected_ticker
)


if not prices.empty:

    st.subheader(
        "Historical Price"
    )

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
        width="stretch",
    )

else:

    st.info(
        f"Historical price data is not available "
        f"for {selected_ticker}."
    )


# ============================================================
# SIGNAL BREAKDOWN
# ============================================================

st.subheader(
    "Signal Breakdown"
)


# ------------------------------------------------------------
# NORMALIZATION FUNCTION
# ------------------------------------------------------------

def normalize_signal(
    series,
    invert=False,
):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    min_value = series.min()
    max_value = series.max()

    if (
        pd.isna(min_value)
        or pd.isna(max_value)
        or max_value == min_value
    ):

        result = pd.Series(
            0.0,
            index=series.index,
        )

    else:

        result = (
            2
            * (
                (series - min_value)
                / (max_value - min_value)
            )
            - 1
        )

    if invert:

        result = -result

    return result


# ------------------------------------------------------------
# NORMALIZE ALL SIGNAL COMPONENTS
# ------------------------------------------------------------

forecast_signal = normalize_signal(
    recommendations["Predicted_Return"]
)

sentiment_signal = normalize_signal(
    recommendations["Sentiment"]
)

volatility_signal = normalize_signal(
    recommendations["Volatility"],
    invert=True,
)

beta_signal = normalize_signal(
    recommendations["Beta"],
    invert=True,
)


# ------------------------------------------------------------
# FIND SELECTED STOCK INDEX
# ------------------------------------------------------------

selected_index = selected.index[0]


# ------------------------------------------------------------
# CREATE SIGNAL DATAFRAME
# ------------------------------------------------------------

signal_df = pd.DataFrame(
    {
        "Signal": [
            "Forecast",
            "Sentiment",
            "Volatility Risk",
            "Beta Risk",
        ],

        "Value": [
            forecast_signal.loc[selected_index],
            sentiment_signal.loc[selected_index],
            volatility_signal.loc[selected_index],
            beta_signal.loc[selected_index],
        ],
    }
)


# ------------------------------------------------------------
# SIGNAL CHART
# ------------------------------------------------------------

fig_signal = px.bar(
    signal_df,
    x="Signal",
    y="Value",
    title=(
        f"{selected_ticker} Signal Components "
        "(Normalized)"
    ),
)


fig_signal.update_layout(
    xaxis_title="Signal Component",
    yaxis_title="Normalized Signal",
    yaxis=dict(
        range=[-1.15, 1.15],
        zeroline=True,
    ),
    hovermode="x unified",
)


st.plotly_chart(
    fig_signal,
    width="stretch",
)


st.caption(
    "Signal components are normalized to a common "
    "−1 to +1 scale for visual comparison. "
    "Higher forecast/sentiment values are bullish; "
    "higher volatility/beta are treated as higher risk."
)


# ============================================================
# RISK DASHBOARD
# ============================================================

st.header(
    "Risk Dashboard"
)

st.subheader(
    f"{selected_ticker} Risk Metrics"
)


risk_prices = load_prices(
    selected_ticker
)


if not risk_prices.empty:

    risk_prices = risk_prices.copy()


    # --------------------------------------------------------
    # DAILY RETURNS
    # --------------------------------------------------------

    risk_prices["Return"] = (
        risk_prices["Close"]
        .pct_change()
    )


    returns = (
        risk_prices["Return"]
        .dropna()
    )


    # --------------------------------------------------------
    # ANNUALIZED VOLATILITY
    # --------------------------------------------------------

    annualized_volatility = (
        returns.std()
        * np.sqrt(252)
    )


    # --------------------------------------------------------
    # BETA
    # --------------------------------------------------------

    benchmark_prices = load_prices(
        "GSPC"
    )


    if not benchmark_prices.empty:

        benchmark_returns = (
            benchmark_prices["Close"]
            .pct_change()
            .dropna()
        )


        aligned_returns = pd.concat(
            [
                returns,
                benchmark_returns,
            ],
            axis=1,
            join="inner",
        )


        aligned_returns.columns = [
            "Stock",
            "Benchmark",
        ]


        beta = (
            aligned_returns["Stock"]
            .cov(
                aligned_returns["Benchmark"]
            )
            /
            aligned_returns["Benchmark"]
            .var()
        )

    else:

        beta = row["Beta"]


    # --------------------------------------------------------
    # ANNUALIZED RETURN
    # --------------------------------------------------------

    annualized_return = (
        returns.mean()
        * 252
    )


    # --------------------------------------------------------
    # SHARPE RATIO
    # --------------------------------------------------------

    if annualized_volatility != 0:

        sharpe_ratio = (
            annualized_return
            / annualized_volatility
        )

    else:

        sharpe_ratio = 0


    # --------------------------------------------------------
    # SORTINO RATIO
    # --------------------------------------------------------

    downside_returns = returns[
        returns < 0
    ]


    if not downside_returns.empty:

        downside_deviation = (
            downside_returns.std()
            * np.sqrt(252)
        )

    else:

        downside_deviation = 0


    if downside_deviation != 0:

        sortino_ratio = (
            annualized_return
            / downside_deviation
        )

    else:

        sortino_ratio = 0


    # --------------------------------------------------------
    # MAXIMUM DRAWDOWN
    # --------------------------------------------------------

    cumulative_returns = (
        1 + returns
    ).cumprod()


    running_max = (
        cumulative_returns
        .cummax()
    )


    drawdown = (
        cumulative_returns
        / running_max
        - 1
    )


    max_drawdown = (
        drawdown.min()
    )


    # --------------------------------------------------------
    # DISPLAY RISK METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )


    col1.metric(
        "Annualized Volatility",
        f"{annualized_volatility * 100:.2f}%",
    )


    col2.metric(
        "Beta",
        f"{beta:.2f}",
    )


    col3.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.3f}",
    )


    col4.metric(
        "Sortino Ratio",
        f"{sortino_ratio:.3f}",
    )


    col5.metric(
        "Maximum Drawdown",
        f"{max_drawdown * 100:.2f}%",
    )


    # --------------------------------------------------------
    # DRAWDOWN CHART
    # --------------------------------------------------------

    drawdown_df = pd.DataFrame(
        {
            "Date": risk_prices.loc[
                returns.index,
                "Date",
            ].values,

            "Drawdown": drawdown.values,
        }
    )


    st.subheader(
        f"{selected_ticker} Historical Drawdown"
    )


    fig_drawdown = px.area(
        drawdown_df,
        x="Date",
        y="Drawdown",
        title=f"{selected_ticker} Drawdown History",
    )


    fig_drawdown.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
    )


    st.plotly_chart(
        fig_drawdown,
        width="stretch",
    )


else:

    st.info(
        f"Risk data is not available "
        f"for {selected_ticker}."
    )


# ============================================================
# DEEP LEARNING MODEL COMPARISON
# ============================================================

st.header(
    "Deep Learning Model Comparison"
)


model_path = os.path.join(
    REPORTS_DIR,
    "dl_model_comparison.csv",
)


if os.path.exists(model_path):

    model_df = pd.read_csv(
        model_path
    )


    st.subheader(
        "LSTM vs GRU vs BiLSTM vs Transformer"
    )


    display_model_df = (
        model_df.copy()
    )


    if "MAE" in display_model_df.columns:

        display_model_df["MAE"] = (
            display_model_df["MAE"]
            .round(6)
        )


    if "RMSE" in display_model_df.columns:

        display_model_df["RMSE"] = (
            display_model_df["RMSE"]
            .round(6)
        )


    if "MAPE" in display_model_df.columns:

        display_model_df["MAPE"] = (
            display_model_df["MAPE"]
            .round(2)
        )


    if "R2" in display_model_df.columns:

        display_model_df["R2"] = (
            display_model_df["R2"]
            .round(4)
        )


    if "Directional_Accuracy" in display_model_df.columns:

        display_model_df[
            "Directional_Accuracy"
        ] = (
            display_model_df[
                "Directional_Accuracy"
            ]
            .round(2)
        )


    st.dataframe(
        display_model_df,
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # MAE CHART
    # --------------------------------------------------------

    if (
        "Model" in model_df.columns
        and "MAE" in model_df.columns
    ):

        fig_model = px.bar(
            model_df,
            x="Model",
            y="MAE",
            title="Model Comparison — MAE",
        )


        fig_model.update_layout(
            xaxis_title="Model",
            yaxis_title="MAE",
        )


        st.plotly_chart(
            fig_model,
            width="stretch",
        )

else:

    st.info(
        "Model comparison results "
        "are not available."
    )


# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

st.header(
    "Portfolio Analytics"
)


portfolio_path = os.path.join(
    REPORTS_DIR,
    "portfolio_walkforward_metrics.csv",
)


if os.path.exists(portfolio_path):

    portfolio_df = pd.read_csv(
        portfolio_path
    )


    # --------------------------------------------------------
    # CLEAN PORTFOLIO NAME
    # --------------------------------------------------------

    if "Unnamed: 0" in portfolio_df.columns:

        portfolio_df = (
            portfolio_df
            .rename(
                columns={
                    "Unnamed: 0":
                    "Portfolio"
                }
            )
        )


    st.subheader(
        "Walk-Forward Portfolio Performance"
    )


    st.dataframe(
        portfolio_df.round(4),
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # SHARPE RATIO
    # --------------------------------------------------------

    if (
        "Sharpe" in portfolio_df.columns
        and "Portfolio"
        in portfolio_df.columns
    ):

        fig_portfolio = px.bar(
            portfolio_df,
            x="Portfolio",
            y="Sharpe",
            title="Walk-Forward Sharpe Ratio",
        )


        fig_portfolio.update_layout(
            xaxis_title="Portfolio",
            yaxis_title="Sharpe Ratio",
        )


        st.plotly_chart(
            fig_portfolio,
            width="stretch",
        )


    # --------------------------------------------------------
    # ANNUALIZED RETURN
    # --------------------------------------------------------

    if (
        "Annualized_Return"
        in portfolio_df.columns
        and "Portfolio"
        in portfolio_df.columns
    ):

        st.subheader(
            "Annualized Return Comparison"
        )


        fig_return = px.bar(
            portfolio_df,
            x="Portfolio",
            y="Annualized_Return",
            title="Walk-Forward Annualized Return",
        )


        fig_return.update_layout(
            xaxis_title="Portfolio",
            yaxis_title="Annualized Return",
        )


        st.plotly_chart(
            fig_return,
            width="stretch",
        )


else:

    st.info(
        "Portfolio walk-forward results "
        "are not available."
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

st.header(
    "Sentiment Analysis"
)


sentiment_path = os.path.join(
    PROCESSED_DIR,
    "finbert_daily_sentiment.csv",
)


if os.path.exists(sentiment_path):

    sentiment_df = pd.read_csv(
        sentiment_path
    )


    sentiment_df[
        "Trading_Date"
    ] = pd.to_datetime(
        sentiment_df["Trading_Date"]
    )


    # --------------------------------------------------------
    # STANDARDIZE TICKER
    # --------------------------------------------------------

    sentiment_df["ticker"] = (
        sentiment_df["ticker"]
        .astype(str)
        .str.upper()
    )


    selected_sentiment = (
        sentiment_df[
            sentiment_df["ticker"]
            == selected_ticker
        ]
        .copy()
    )


    if not selected_sentiment.empty:

        selected_sentiment = (
            selected_sentiment
            .sort_values(
                "Trading_Date"
            )
        )


        st.subheader(
            f"{selected_ticker} FinBERT Sentiment"
        )


        # ----------------------------------------------------
        # LATEST SENTIMENT METRICS
        # ----------------------------------------------------

        latest_sentiment = (
            selected_sentiment
            .iloc[-1]
        )


        sentiment_mean = (
            latest_sentiment[
                "FinBERT_Sentiment_Mean"
            ]
        )


        news_volume = (
            latest_sentiment[
                "FinBERT_News_Volume"
            ]
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "Latest Sentiment",
            f"{sentiment_mean:.3f}",
        )


        col2.metric(
            "News Volume",
            int(news_volume),
        )


        # ----------------------------------------------------
        # SENTIMENT HISTORY
        # ----------------------------------------------------

        st.subheader(
            f"{selected_ticker} Daily FinBERT Sentiment"
        )


        fig_sentiment = px.line(
            selected_sentiment,
            x="Trading_Date",
            y="FinBERT_Sentiment_Mean",
            title=(
                f"{selected_ticker} Daily "
                "FinBERT Sentiment"
            ),
        )


        fig_sentiment.add_hline(
            y=0,
            line_dash="dash",
        )


        fig_sentiment.update_layout(
            xaxis_title="Trading Date",
            yaxis_title="Sentiment Score",
            hovermode="x unified",
        )


        st.plotly_chart(
            fig_sentiment,
            width="stretch",
        )


        # ----------------------------------------------------
        # NEWS VOLUME
        # ----------------------------------------------------

        if (
            "FinBERT_News_Volume"
            in selected_sentiment.columns
        ):

            st.subheader(
                f"{selected_ticker} Daily News Volume"
            )


            fig_news = px.bar(
                selected_sentiment,
                x="Trading_Date",
                y="FinBERT_News_Volume",
                title=(
                    f"{selected_ticker} "
                    "Daily News Volume"
                ),
            )


            fig_news.update_layout(
                xaxis_title="Trading Date",
                yaxis_title=(
                    "Number of News Articles"
                ),
            )


            st.plotly_chart(
                fig_news,
                width="stretch",
            )


    else:

        st.info(
            f"No historical FinBERT sentiment "
            f"data is available for "
            f"{selected_ticker}."
        )


else:

    st.info(
        "FinBERT sentiment data "
        "is not available."
    )


# ============================================================
# FINAL DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Northgate AI Stock Predictor uses historical market data, "
    "machine-learning forecasts, sentiment signals and risk metrics. "
    "Past performance does not guarantee future results."
)