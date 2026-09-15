import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
)

REPORTS_DIR = os.path.join(
    ROOT_DIR,
    "reports",
)

sys.path.append(
    os.path.join(
        ROOT_DIR,
        "src",
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

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

FORECAST_FILE = os.path.join(
    REPORTS_DIR,
    "transformer_holdout_forecasts.csv",
)

SENTIMENT_FILE = os.path.join(
    PROCESSED_DIR,
    "finbert_daily_sentiment.csv",
)

BENCHMARK_FILE = os.path.join(
    PROCESSED_DIR,
    "GSPC.csv",
)


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data(ticker):

    path = os.path.join(
        PROCESSED_DIR,
        f"{ticker}.csv",
    )

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    return (
        df
        .sort_values("Date")
        .set_index("Date")
    )


# ============================================================
# LOAD SENTIMENT
# ============================================================

def load_sentiment():

    if not os.path.exists(
        SENTIMENT_FILE
    ):
        return pd.DataFrame()

    df = pd.read_csv(
        SENTIMENT_FILE
    )

    df["Trading_Date"] = pd.to_datetime(
        df["Trading_Date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
    )

    return df


def get_sentiment(
    sentiment_df,
    ticker,
    dates,
):

    if sentiment_df.empty:

        return pd.Series(
            0.0,
            index=dates,
        )

    result = (
        sentiment_df[
            sentiment_df["ticker"]
            == ticker
        ]
        [
            [
                "Trading_Date",
                "FinBERT_Sentiment_Mean",
            ]
        ]
        .drop_duplicates(
            "Trading_Date"
        )
        .set_index(
            "Trading_Date"
        )[
            "FinBERT_Sentiment_Mean"
        ]
    )

    return (
        result
        .reindex(dates)
        .fillna(0.0)
    )


# ============================================================
# CAUSAL RISK FEATURES
# ============================================================

def calculate_risk_features(
    stock,
    benchmark,
    dates,
):

    stock_returns = (
        stock["Close"]
        .pct_change()
    )

    benchmark_returns = (
        benchmark["Close"]
        .pct_change()
    )

    volatility = (
        stock_returns
        .rolling(
            60,
            min_periods=60,
        )
        .std()
        * np.sqrt(252)
    )

    running_max = (
        stock["Close"]
        .cummax()
    )

    drawdown = (
        stock["Close"]
        / running_max
        - 1.0
    )

    max_drawdown = (
        drawdown
        .rolling(
            252,
            min_periods=252,
        )
        .min()
    )

    covariance = (
        stock_returns
        .rolling(
            60,
            min_periods=60,
        )
        .cov(
            benchmark_returns
        )
    )

    benchmark_variance = (
        benchmark_returns
        .rolling(
            60,
            min_periods=60,
        )
        .var()
    )

    beta = (
        covariance
        / benchmark_variance
    )

    return (
        volatility.reindex(dates),
        max_drawdown.reindex(dates),
        beta.reindex(dates),
    )


# ============================================================
# CAUSAL EXPANDING NORMALIZATION
# ============================================================

def expanding_minmax(series):

    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    for i in range(
        len(series)
    ):

        current = series.iloc[i]

        if pd.isna(current):
            continue

        history = (
            series
            .iloc[: i + 1]
            .dropna()
        )

        if len(history) < 2:
            output.iloc[i] = 0.0
            continue

        low = history.min()
        high = history.max()

        if high == low:

            output.iloc[i] = 0.0

        else:

            output.iloc[i] = np.clip(
                (
                    (
                        current - low
                    )
                    / (
                        high - low
                    )
                )
                * 2.0
                - 1.0,
                -1.0,
                1.0,
            )

    return output


# ============================================================
# EVALUATE SIGNAL
# ============================================================

def evaluate_signal(
    signal,
    actual_returns,
    name,
    ticker,
):

    signal = pd.Series(
        signal,
        index=actual_returns.index,
    )

    recommendation = np.select(
        [
            signal >= 0.20,
            signal <= -0.20,
        ],
        [
            "BUY",
            "SELL",
        ],
        default="HOLD",
    )

    directional = (
        recommendation
        != "HOLD"
    )

    if directional.sum() == 0:

        hit_rate = np.nan

    else:

        correct = (
            (
                (
                    recommendation
                    == "BUY"
                )
                &
                (
                    actual_returns
                    > 0
                )
            )
            |
            (
                (
                    recommendation
                    == "SELL"
                )
                &
                (
                    actual_returns
                    < 0
                )
            )
        )

        hit_rate = (
            correct[
                directional
            ].mean()
            * 100
        )

    strategy_return = np.select(
        [
            recommendation
            == "BUY",

            recommendation
            == "SELL",
        ],
        [
            actual_returns.to_numpy(),

            -actual_returns.to_numpy(),
        ],
        default=0.0,
    )

    equity = np.cumprod(
        1.0 + strategy_return
    )

    total_return = (
        equity[-1] - 1.0
    ) * 100

    return {
        "Ticker": ticker,
        "Component": name,
        "Observations": len(
            actual_returns
        ),
        "Directional_Calls": int(
            directional.sum()
        ),
        "BUY_Count": int(
            (
                recommendation
                == "BUY"
            ).sum()
        ),
        "HOLD_Count": int(
            (
                recommendation
                == "HOLD"
            ).sum()
        ),
        "SELL_Count": int(
            (
                recommendation
                == "SELL"
            ).sum()
        ),
        "Hit_Rate_%": hit_rate,
        "Strategy_Return_%":
            total_return,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "NORTHGATE RECOMMENDATION "
        "COMPONENT ABLATION"
    )
    print("=" * 70)

    forecasts = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["Date"],
    )

    sentiment_df = (
        load_sentiment()
    )

    benchmark = pd.read_csv(
        BENCHMARK_FILE,
        parse_dates=["Date"],
    )

    benchmark = (
        benchmark
        .sort_values("Date")
        .set_index("Date")
    )

    print(
        f"Transformer forecasts: "
        f"{len(forecasts)} rows"
    )

    print(
        f"Sentiment rows: "
        f"{len(sentiment_df)}"
    )

    all_results = []

    for ticker in TICKERS:

        print()
        print("-" * 70)
        print(
            f"Processing {ticker}"
        )

        data = (
            forecasts[
                forecasts["Ticker"]
                == ticker
            ]
            .sort_values("Date")
            .copy()
        )

        if data.empty:
            continue

        data = data.set_index(
            "Date"
        )

        dates = data.index

        stock = load_price_data(
            ticker
        )

        (
            volatility,
            max_drawdown,
            beta,
        ) = calculate_risk_features(
            stock,
            benchmark,
            dates,
        )

        sentiment = get_sentiment(
            sentiment_df,
            ticker,
            dates,
        )

        # ----------------------------------------------------
        # Causal component signals
        # ----------------------------------------------------

        forecast_signal = (
            expanding_minmax(
                data[
                    "Predicted_Return"
                ]
            )
        )

        sentiment_signal = (
            expanding_minmax(
                sentiment
            )
        )

        # Risk features themselves are causal.
        # Normalize each using only past/current observations.

        volatility_signal = (
            expanding_minmax(
                volatility
            )
        )

        drawdown_signal = (
            expanding_minmax(
                max_drawdown
            )
        )

        beta_signal = (
            expanding_minmax(
                beta
            )
        )

        # Convert risk components to positive risk scores.

        volatility_risk = (
            volatility_signal + 1.0
        ) / 2.0

        drawdown_risk = (
            1.0
            - (
                drawdown_signal
                + 1.0
            ) / 2.0
        )

        beta_risk = (
            beta_signal + 1.0
        ) / 2.0

        risk_score = (
            0.40
            * volatility_risk
            +
            0.30
            * drawdown_risk
            +
            0.30
            * beta_risk
        )

        # ----------------------------------------------------
        # Combined production signal
        # ----------------------------------------------------

        combined_signal = (
            0.50
            * forecast_signal
            +
            0.30
            * sentiment_signal
            -
            0.20
            * risk_score
        )

        valid = pd.DataFrame(
            {
                "Actual_Return":
                    data[
                        "Actual_Return"
                    ],

                "Forecast":
                    forecast_signal,

                "Sentiment":
                    sentiment_signal,

                "Risk":
                    risk_score,

                "Combined":
                    combined_signal,
            }
        ).dropna()

        if valid.empty:
            continue

        # ----------------------------------------------------
        # Evaluate components
        # ----------------------------------------------------

        all_results.append(
            evaluate_signal(
                valid["Forecast"],
                valid[
                    "Actual_Return"
                ],
                "Forecast Only",
                ticker,
            )
        )

        all_results.append(
            evaluate_signal(
                valid["Sentiment"],
                valid[
                    "Actual_Return"
                ],
                "Sentiment Only",
                ticker,
            )
        )

        # Risk is a penalty.
        # For a standalone risk test, lower risk = BUY signal.
        risk_signal = (
            1.0
            - 2.0
            * valid["Risk"]
        )

        all_results.append(
            evaluate_signal(
                risk_signal,
                valid[
                    "Actual_Return"
                ],
                "Risk Only",
                ticker,
            )
        )

        all_results.append(
            evaluate_signal(
                valid["Combined"],
                valid[
                    "Actual_Return"
                ],
                "Combined",
                ticker,
            )
        )

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_path = os.path.join(
        REPORTS_DIR,
        "recommendation_ablation_details.csv",
    )

    results.to_csv(
        detailed_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results
        .groupby("Component")
        .agg(
            Observations=(
                "Observations",
                "sum",
            ),
            Directional_Calls=(
                "Directional_Calls",
                "sum",
            ),
            BUY_Count=(
                "BUY_Count",
                "sum",
            ),
            HOLD_Count=(
                "HOLD_Count",
                "sum",
            ),
            SELL_Count=(
                "SELL_Count",
                "sum",
            ),
            Average_Hit_Rate_pct=(
                "Hit_Rate_%",
                "mean",
            ),
            Average_Strategy_Return_pct=(
                "Strategy_Return_%",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = os.path.join(
        REPORTS_DIR,
        "recommendation_ablation.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "ABLATION SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved detailed results:"
    )
    print(detailed_path)

    print()
    print(
        "Saved summary:"
    )
    print(summary_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()