import os

import numpy as np
import pandas as pd

from recommend import generate_recommendations


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

TICKERS = [
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "NVDA",
    "KO",
    "CAT",
    "HD",
]


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data():
    """
    Load processed stock prices.
    """

    prices = {}

    for ticker in TICKERS:

        path = os.path.join(
            PROCESSED_DIR,
            f"{ticker}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_csv(
            path,
            parse_dates=["Date"],
        )

        df = (
            df.set_index("Date")
            .sort_index()
        )

        prices[ticker] = df["Close"]

    return pd.DataFrame(prices).dropna()


# ============================================================
# CALCULATE RISK METRICS
# ============================================================

def calculate_risk_metrics(prices):
    """
    Calculate annualized volatility, maximum drawdown
    and beta for each stock.
    """

    returns = prices.pct_change().dropna()

    benchmark_path = os.path.join(
        PROCESSED_DIR,
        "GSPC.csv",
    )

    benchmark = pd.read_csv(
        benchmark_path,
        parse_dates=["Date"],
    )

    benchmark = (
        benchmark.set_index("Date")
        .sort_index()["Close"]
    )

    benchmark_returns = (
        benchmark
        .pct_change()
        .dropna()
    )

    rows = []

    for ticker in prices.columns:

        ticker_returns = (
            returns[ticker]
            .dropna()
        )

        volatility = (
            ticker_returns.std()
            * np.sqrt(252)
        )

        wealth = (
            1 + ticker_returns
        ).cumprod()

        running_max = (
            wealth.cummax()
        )

        drawdown = (
            wealth / running_max
        ) - 1

        max_drawdown = (
            drawdown.min()
        )

        aligned = pd.concat(
            [
                ticker_returns.rename(
                    "stock"
                ),
                benchmark_returns.rename(
                    "benchmark"
                ),
            ],
            axis=1,
        ).dropna()

        benchmark_variance = (
            aligned["benchmark"]
            .var()
        )

        if benchmark_variance == 0:
            beta = np.nan
        else:
            beta = (
                aligned["stock"]
                .cov(
                    aligned["benchmark"]
                )
                / benchmark_variance
            )

        rows.append(
            {
                "Ticker": ticker,
                "Volatility": volatility,
                "Max_Drawdown": max_drawdown,
                "Beta": beta,
            }
        )

    return pd.DataFrame(
        rows
    ).set_index("Ticker")


# ============================================================
# LOAD SENTIMENT
# ============================================================

def load_sentiment():
    """
    Load ticker-specific daily FinBERT sentiment.

    Uses the most recent available sentiment observation
    for each stock.
    """

    path = os.path.join(
        PROCESSED_DIR,
        "sentiment_features.csv",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing sentiment file: {path}"
        )

    sentiment = pd.read_csv(
        path
    )

    required_columns = {
        "ticker",
        "Trading_Date",
        "FinBERT_Sentiment_Mean",
    }

    missing = (
        required_columns
        - set(sentiment.columns)
    )

    if missing:
        raise ValueError(
            "Missing sentiment columns: "
            + str(missing)
        )

    sentiment["Trading_Date"] = (
        pd.to_datetime(
            sentiment["Trading_Date"]
        )
    )

    sentiment = (
        sentiment
        .sort_values(
            ["ticker", "Trading_Date"]
        )
        .groupby("ticker")
        .tail(1)
    )

    sentiment = sentiment.set_index(
        "ticker"
    )

    return sentiment[
        "FinBERT_Sentiment_Mean"
    ]


# ============================================================
# LOAD FORECASTS
# ============================================================

def load_forecasts():
    """
    Load the latest available model forecast.

    The current modeling pipeline does not yet save a
    standardized production forecast file.

    Therefore this function first checks whether one exists.
    """

    possible_files = [
        "model_forecasts.csv",
        "latest_forecasts.csv",
    ]

    forecast_path = None

    for filename in possible_files:

        path = os.path.join(
            REPORTS_DIR,
            filename,
        )

        if os.path.exists(path):
            forecast_path = path
            break

    if forecast_path is None:
        raise FileNotFoundError(
            "No forecast file found.\n"
            "Expected one of:\n"
            "  reports/model_forecasts.csv\n"
            "  reports/latest_forecasts.csv"
        )

    forecasts = pd.read_csv(
        forecast_path
    )

    if "Ticker" not in forecasts.columns:
        raise ValueError(
            "Forecast file must contain "
            "'Ticker' column."
        )

    forecast_column = None

    for column in [
        "Predicted_Return",
        "Prediction",
        "Forecast",
        "Predicted",
    ]:

        if column in forecasts.columns:
            forecast_column = column
            break

    if forecast_column is None:
        raise ValueError(
            "Forecast file must contain a prediction "
            "column such as Predicted_Return."
        )

    forecasts = forecasts[
        ["Ticker", forecast_column]
    ].copy()

    forecasts = forecasts.rename(
        columns={
            forecast_column:
                "Predicted_Return"
        }
    )

    forecasts = forecasts.drop_duplicates(
        "Ticker",
        keep="last",
    )

    forecasts = forecasts.set_index(
        "Ticker"
    )

    return forecasts[
        "Predicted_Return"
    ]


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendations():

    print(
        "=" * 70
    )

    print(
        "NORTHGATE REAL RECOMMENDATION PIPELINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Prices
    # --------------------------------------------------------

    prices = load_price_data()

    print(
        f"Loaded price data: "
        f"{prices.shape}"
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = calculate_risk_metrics(
        prices
    )

    print(
        f"Calculated risk metrics for "
        f"{len(risk)} stocks."
    )

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    sentiment = load_sentiment()

    print(
        f"Loaded sentiment for "
        f"{len(sentiment)} stocks."
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    forecasts = load_forecasts()

    print(
        f"Loaded forecasts for "
        f"{len(forecasts)} stocks."
    )

    # --------------------------------------------------------
    # Align stocks
    # --------------------------------------------------------

    available = [
        ticker
        for ticker in TICKERS
        if (
            ticker in forecasts.index
            and ticker in sentiment.index
            and ticker in risk.index
        )
    ]

    if not available:
        raise ValueError(
            "No stocks have complete "
            "forecast + sentiment + risk data."
        )

    print(
        f"Stocks with complete data: "
        f"{len(available)}"
    )

    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    forecast_series = forecasts.reindex(
        available
    )

    sentiment_series = sentiment.reindex(
        available
    )

    volatility_series = risk.loc[
        available,
        "Volatility",
    ]

    drawdown_series = risk.loc[
        available,
        "Max_Drawdown",
    ]

    beta_series = risk.loc[
        available,
        "Beta",
    ]

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            forecast_returns=forecast_series,
            sentiment_scores=sentiment_series,
            volatility=volatility_series,
            max_drawdown=drawdown_series,
            beta=beta_series,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        REPORTS_DIR,
        "stock_recommendations.csv",
    )

    recommendations.to_csv(
        output_path
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL STOCK RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "Predicted_Return",
        "Sentiment",
        "Volatility",
        "Beta",
        "Composite_Score",
        "Recommendation",
    ]

    print(
        recommendations[
            display_columns
        ].round(4)
    )

    print()
    print(
        "Recommendation counts:"
    )

    print(
        recommendations[
            "Recommendation"
        ].value_counts()
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    return recommendations


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_recommendations()