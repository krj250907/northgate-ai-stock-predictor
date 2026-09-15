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

from recommend import (
    FORECAST_WEIGHT,
    SENTIMENT_WEIGHT,
    RISK_WEIGHT,
    SELL_THRESHOLD,
    BUY_THRESHOLD,
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

BENCHMARK_FILE = os.path.join(
    PROCESSED_DIR,
    "GSPC.csv",
)

SENTIMENT_FILE = os.path.join(
    PROCESSED_DIR,
    "finbert_daily_sentiment.csv",
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

    df = (
        df
        .sort_values("Date")
        .set_index("Date")
    )

    return df


# ============================================================
# LOAD SENTIMENT
# ============================================================

def load_sentiment_data():

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


def get_sentiment_series(
    sentiment_df,
    ticker,
    dates,
):

    if sentiment_df.empty:

        return pd.Series(
            0.0,
            index=dates,
        )

    ticker_data = (
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
        )
        [
            "FinBERT_Sentiment_Mean"
        ]
    )

    return (
        ticker_data
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

    # --------------------------------------------------------
    # 60-day annualized volatility
    #
    # At date t this only uses returns up to t.
    # --------------------------------------------------------

    volatility = (
        stock_returns
        .rolling(
            60,
            min_periods=60,
        )
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # Historical drawdown
    #
    # Running maximum avoids looking into the future.
    # --------------------------------------------------------

    running_max = (
        stock["Close"]
        .cummax()
    )

    drawdown = (
        stock["Close"]
        / running_max
        - 1.0
    )

    # --------------------------------------------------------
    # 252-day rolling minimum drawdown
    #
    # Only uses current/past observations.
    # --------------------------------------------------------

    max_drawdown = (
        drawdown
        .rolling(
            252,
            min_periods=252,
        )
        .min()
    )

    # --------------------------------------------------------
    # 60-day rolling beta
    # --------------------------------------------------------

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
# CAUSAL NORMALIZATION
# ============================================================

def causal_min_max(
    series,
    min_value,
    max_value,
):

    denominator = (
        max_value
        - min_value
    )

    if denominator == 0:
        return 0.0

    value = (
        (series - min_value)
        / denominator
    )

    value = np.clip(
        value,
        0.0,
        1.0,
    )

    return (
        value * 2.0
        - 1.0
    )


def calculate_causal_risk_signal(
    volatility,
    max_drawdown,
    beta,
):

    result = pd.DataFrame(
        {
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "beta": beta,
        }
    )

    result["risk_signal"] = np.nan

    # --------------------------------------------------------
    # Expanding historical normalization
    #
    # Each date is normalized using ONLY observations
    # available before or at that date.
    # --------------------------------------------------------

    for i in range(len(result)):

        history = result.iloc[: i + 1]

        vol = history[
            "volatility"
        ].dropna()

        dd = history[
            "max_drawdown"
        ].dropna()

        b = history[
            "beta"
        ].dropna()

        if (
            len(vol) < 2
            or len(dd) < 2
            or len(b) < 2
        ):
            continue

        current = result.iloc[i]

        if (
            pd.isna(
                current["volatility"]
            )
            or pd.isna(
                current["max_drawdown"]
            )
            or pd.isna(
                current["beta"]
            )
        ):
            continue

        vol_min = vol.min()
        vol_max = vol.max()

        dd_min = dd.min()
        dd_max = dd.max()

        beta_min = b.min()
        beta_max = b.max()

        vol_signal = (
            causal_min_max(
                current["volatility"],
                vol_min,
                vol_max,
            )
        )

        # More negative drawdown = higher risk.
        # Convert so higher value means higher risk.

        dd_normalized = (
            causal_min_max(
                current["max_drawdown"],
                dd_min,
                dd_max,
            )
        )

        dd_risk = (
            1.0
            - (
                dd_normalized
                + 1.0
            )
            / 2.0
        )

        beta_normalized = (
            causal_min_max(
                current["beta"],
                beta_min,
                beta_max,
            )
        )

        beta_risk = (
            (
                beta_normalized
                + 1.0
            )
            / 2.0
        )

        vol_risk = (
            (
                vol_signal
                + 1.0
            )
            / 2.0
        )

        risk_score = (
            0.40 * vol_risk
            + 0.30 * dd_risk
            + 0.30 * beta_risk
        )

        result.iloc[
            i,
            result.columns.get_loc(
                "risk_signal"
            ),
        ] = risk_score

    return result[
        "risk_signal"
    ]


# ============================================================
# FORECAST SIGNAL
# ============================================================

def calculate_forecast_signal(
    forecast_returns
):

    # Causal cross-sectional normalization is not appropriate
    # here because this backtest is performed one stock at a
    # time. Use a fixed bounded transformation based on the
    # historical forecast distribution available through t.

    result = pd.Series(
        np.nan,
        index=forecast_returns.index,
        dtype=float,
    )

    for i in range(
        len(forecast_returns)
    ):

        history = (
            forecast_returns
            .iloc[: i + 1]
            .dropna()
        )

        if len(history) < 2:
            continue

        current = forecast_returns.iloc[i]

        if pd.isna(current):
            continue

        lower = history.min()
        upper = history.max()

        if upper == lower:
            result.iloc[i] = 0.0
        else:
            result.iloc[i] = np.clip(
                (
                    (
                        current
                        - lower
                    )
                    / (
                        upper
                        - lower
                    )
                )
                * 2.0
                - 1.0,
                -1.0,
                1.0,
            )

    return result


# ============================================================
# SENTIMENT SIGNAL
# ============================================================

def calculate_sentiment_signal(
    sentiment
):

    result = pd.Series(
        np.nan,
        index=sentiment.index,
        dtype=float,
    )

    for i in range(
        len(sentiment)
    ):

        history = (
            sentiment
            .iloc[: i + 1]
            .dropna()
        )

        current = sentiment.iloc[i]

        if pd.isna(current):
            continue

        if len(history) < 2:
            result.iloc[i] = 0.0
            continue

        lower = history.min()
        upper = history.max()

        if upper == lower:
            result.iloc[i] = 0.0
        else:
            result.iloc[i] = np.clip(
                (
                    (
                        current
                        - lower
                    )
                    / (
                        upper
                        - lower
                    )
                )
                * 2.0
                - 1.0,
                -1.0,
                1.0,
            )

    return result


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_causal_recommendations(
    forecast_returns,
    sentiment,
    volatility,
    max_drawdown,
    beta,
):

    forecast_signal = (
        calculate_forecast_signal(
            forecast_returns
        )
    )

    sentiment_signal = (
        calculate_sentiment_signal(
            sentiment
        )
    )

    risk_signal = (
        calculate_causal_risk_signal(
            volatility,
            max_drawdown,
            beta,
        )
    )

    result = pd.DataFrame(
        {
            "Forecast_Return":
                forecast_returns,

            "Sentiment":
                sentiment,

            "Volatility":
                volatility,

            "Max_Drawdown":
                max_drawdown,

            "Beta":
                beta,

            "Forecast_Signal":
                forecast_signal,

            "Sentiment_Signal":
                sentiment_signal,

            "Risk_Signal":
                risk_signal,
        }
    )

    result["Composite_Score"] = (
        FORECAST_WEIGHT
        * result["Forecast_Signal"]
        +
        SENTIMENT_WEIGHT
        * result["Sentiment_Signal"]
        -
        RISK_WEIGHT
        * result["Risk_Signal"]
    )

    result["Recommendation"] = np.select(
        [
            result["Composite_Score"]
            <= SELL_THRESHOLD,

            result["Composite_Score"]
            >= BUY_THRESHOLD,
        ],
        [
            "SELL",
            "BUY",
        ],
        default="HOLD",
    )

    return result


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "NORTHGATE TRANSFORMER "
        "RECOMMENDATION BACKTEST"
    )
    print("=" * 70)

    if not os.path.exists(
        FORECAST_FILE
    ):
        raise FileNotFoundError(
            f"Missing forecast file: "
            f"{FORECAST_FILE}"
        )

    forecasts = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["Date"],
    )

    print(
        f"Historical forecasts: "
        f"{len(forecasts)} rows"
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

    sentiment_df = (
        load_sentiment_data()
    )

    if sentiment_df.empty:
        print(
            "WARNING: Sentiment data unavailable."
        )
    else:
        print(
            f"Sentiment rows: "
            f"{len(sentiment_df)}"
        )

    all_results = []
    summary = []

    # ========================================================
    # EACH STOCK
    # ========================================================

    for ticker in TICKERS:

        print()
        print("-" * 70)
        print(
            f"Processing {ticker}"
        )

        ticker_forecasts = (
            forecasts[
                forecasts["Ticker"]
                == ticker
            ]
            .sort_values("Date")
            .copy()
        )

        if ticker_forecasts.empty:
            print(
                f"Skipping {ticker}: "
                "no forecasts."
            )
            continue

        ticker_forecasts = (
            ticker_forecasts
            .set_index("Date")
        )

        dates = ticker_forecasts.index

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

        sentiment = (
            get_sentiment_series(
                sentiment_df,
                ticker,
                dates,
            )
        )

        recommendations = (
            generate_causal_recommendations(
                forecast_returns=
                    ticker_forecasts[
                        "Predicted_Return"
                    ],

                sentiment=
                    sentiment,

                volatility=
                    volatility,

                max_drawdown=
                    max_drawdown,

                beta=
                    beta,
            )
        )

        recommendations["Date"] = dates

        recommendations[
            "Actual_Return"
        ] = ticker_forecasts[
            "Actual_Return"
        ].values

        recommendations[
            "Ticker"
        ] = ticker

        recommendations = (
            recommendations
            .dropna(
                subset=[
                    "Actual_Return",
                    "Composite_Score",
                ]
            )
            .copy()
        )

        directional = (
            recommendations[
                recommendations[
                    "Recommendation"
                ].isin(
                    [
                        "BUY",
                        "SELL",
                    ]
                )
            ]
            .copy()
        )

        if directional.empty:
            hit_rate = np.nan
        else:

            correct = (
                (
                    (
                        directional[
                            "Recommendation"
                        ]
                        == "BUY"
                    )
                    &
                    (
                        directional[
                            "Actual_Return"
                        ]
                        > 0
                    )
                )
                |
                (
                    (
                        directional[
                            "Recommendation"
                        ]
                        == "SELL"
                    )
                    &
                    (
                        directional[
                            "Actual_Return"
                        ]
                        < 0
                    )
                )
            )

            hit_rate = (
                correct.mean()
                * 100
            )

        baseline_correct = (
            recommendations[
                "Actual_Return"
            ]
            > 0
        )

        baseline_hit_rate = (
            baseline_correct.mean()
            * 100
        )

        recommendations[
            "Strategy_Return"
        ] = np.select(
            [
                recommendations[
                    "Recommendation"
                ]
                == "BUY",

                recommendations[
                    "Recommendation"
                ]
                == "SELL",
            ],
            [
                recommendations[
                    "Actual_Return"
                ],

                -recommendations[
                    "Actual_Return"
                ],
            ],
            default=0.0,
        )

        recommendations[
            "Strategy_Equity"
        ] = (
            1.0
            + recommendations[
                "Strategy_Return"
            ]
        ).cumprod()

        recommendations[
            "BuyHold_Equity"
        ] = (
            1.0
            + recommendations[
                "Actual_Return"
            ]
        ).cumprod()

        strategy_total_return = (
            recommendations[
                "Strategy_Equity"
            ].iloc[-1]
            - 1
        )

        buyhold_total_return = (
            recommendations[
                "BuyHold_Equity"
            ].iloc[-1]
            - 1
        )

        all_results.append(
            recommendations
        )

        summary.append(
            {
                "Ticker": ticker,

                "Observations":
                    len(
                        recommendations
                    ),

                "Directional_Calls":
                    len(
                        directional
                    ),

                "BUY_Count":
                    int(
                        (
                            recommendations[
                                "Recommendation"
                            ]
                            == "BUY"
                        ).sum()
                    ),

                "HOLD_Count":
                    int(
                        (
                            recommendations[
                                "Recommendation"
                            ]
                            == "HOLD"
                        ).sum()
                    ),

                "SELL_Count":
                    int(
                        (
                            recommendations[
                                "Recommendation"
                            ]
                            == "SELL"
                        ).sum()
                    ),

                "Recommendation_Hit_Rate_%":
                    hit_rate,

                "BuyHold_Hit_Rate_%":
                    baseline_hit_rate,

                "Strategy_Total_Return_%":
                    strategy_total_return
                    * 100,

                "BuyHold_Total_Return_%":
                    buyhold_total_return
                    * 100,
            }
        )

        print(
            f"Recommendation hit rate: "
            f"{hit_rate:.2f}%"
        )

        print(
            f"Buy-and-hold hit rate: "
            f"{baseline_hit_rate:.2f}%"
        )

        print(
            f"Strategy return: "
            f"{strategy_total_return * 100:.2f}%"
        )

        print(
            f"Buy-and-hold return: "
            f"{buyhold_total_return * 100:.2f}%"
        )

    # ========================================================
    # SAVE DETAILS
    # ========================================================

    if not all_results:
        raise RuntimeError(
            "No recommendation "
            "backtest results generated."
        )

    detailed = pd.concat(
        all_results,
        ignore_index=True,
    )

    detailed = detailed.sort_values(
        [
            "Ticker",
            "Date",
        ]
    )

    detailed_path = os.path.join(
        REPORTS_DIR,
        "recommendation_transformer_backtest_causal_details.csv",
    )

    detailed.to_csv(
        detailed_path,
        index=False,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    summary_path = os.path.join(
        REPORTS_DIR,
        "recommendation_transformer_backtest_causal.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "LEAKAGE-SAFE TRANSFORMER "
        "RECOMMENDATION BACKTEST SUMMARY"
    )
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Average recommendation hit rate: "
        f"{summary_df['Recommendation_Hit_Rate_%'].mean():.2f}%"
    )

    print(
        "Average buy-and-hold hit rate: "
        f"{summary_df['BuyHold_Hit_Rate_%'].mean():.2f}%"
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