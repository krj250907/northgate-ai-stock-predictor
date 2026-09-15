import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
    os.path.join(ROOT_DIR, "src")
)

from recommend import (
    generate_recommendations,
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

LOOKBACK_DAYS = 30


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data(ticker):

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

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD HISTORICAL FORECAST
# ============================================================

def build_historical_forecast(
    prices,
    index,
):
    """
    Simple leakage-free baseline forecast.

    At date t, the forecast uses the
    previous day's return.

    This is intentionally conservative.
    """

    close = prices["Close"]

    returns = (
        close
        .pct_change()
    )

    forecast = (
        returns
        .shift(1)
    )

    return forecast.loc[index]


# ============================================================
# HISTORICAL RISK ESTIMATES
# ============================================================

def calculate_historical_risk(
    prices,
    index,
):

    returns = (
        prices["Close"]
        .pct_change()
    )

    volatility = (
        returns
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    rolling_max = (
        prices["Close"]
        .rolling(252)
        .max()
    )

    drawdown = (
        prices["Close"]
        / rolling_max
        - 1
    )

    max_drawdown = (
        drawdown
        .rolling(252)
        .min()
    )

    return (
        volatility.loc[index],
        max_drawdown.loc[index],
    )


# ============================================================
# HISTORICAL BETA
# ============================================================

def calculate_beta(
    stock_prices,
    benchmark_prices,
    index,
):

    stock_returns = (
        stock_prices["Close"]
        .pct_change()
    )

    benchmark_returns = (
        benchmark_prices["Close"]
        .pct_change()
    )

    combined = pd.concat(
        [
            stock_returns.rename(
                "Stock"
            ),
            benchmark_returns.rename(
                "Benchmark"
            ),
        ],
        axis=1,
    )

    rolling_cov = (
        combined["Stock"]
        .rolling(60)
        .cov(
            combined["Benchmark"]
        )
    )

    rolling_var = (
        combined["Benchmark"]
        .rolling(60)
        .var()
    )

    beta = (
        rolling_cov
        / rolling_var
    )

    return beta.loc[index]


# ============================================================
# SENTIMENT DATA
# ============================================================

def load_sentiment():

    path = os.path.join(
        PROCESSED_DIR,
        "finbert_daily_sentiment.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(
        path
    )

    df["Trading_Date"] = (
        pd.to_datetime(
            df["Trading_Date"]
        )
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
    )

    return df


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NORTHGATE RECOMMENDATION BACKTEST")
    print("=" * 70)


    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    benchmark = load_price_data(
        "GSPC"
    )

    if benchmark.empty:

        raise FileNotFoundError(
            "Benchmark file GSPC.csv not found."
        )


    benchmark = benchmark[
        [
            "Date",
            "Close",
        ]
    ].copy()


    # --------------------------------------------------------
    # Load sentiment
    # --------------------------------------------------------

    sentiment_df = load_sentiment()


    if sentiment_df.empty:

        print(
            "WARNING: Sentiment data not found."
        )


    results = []


    # ========================================================
    # PROCESS EACH STOCK
    # ========================================================

    for ticker in TICKERS:

        print(
            f"\nProcessing {ticker}..."
        )


        prices = load_price_data(
            ticker
        )


        if prices.empty:

            print(
                f"Skipping {ticker}: "
                "price data unavailable."
            )

            continue


        prices = prices[
            [
                "Date",
                "Close",
            ]
        ].copy()


        prices = prices.sort_values(
            "Date"
        )


        # ----------------------------------------------------
        # Merge benchmark
        # ----------------------------------------------------

        merged = prices.merge(
            benchmark,
            on="Date",
            how="inner",
            suffixes=(
                "_Stock",
                "_Benchmark",
            ),
        )


        # ----------------------------------------------------
        # Actual next-day return
        # ----------------------------------------------------

        merged[
            "Actual_Return"
        ] = (
            merged["Close_Stock"]
            .pct_change()
            .shift(-1)
        )


        # ----------------------------------------------------
        # Forecast available at t
        # ----------------------------------------------------

        stock_returns = (
            merged["Close_Stock"]
            .pct_change()
        )


        merged[
            "Forecast"
        ] = stock_returns.shift(1)


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        merged[
            "Volatility"
        ] = (
            stock_returns
            .rolling(60)
            .std()
            * np.sqrt(252)
        )


        rolling_max = (
            merged["Close_Stock"]
            .rolling(252)
            .max()
        )


        merged[
            "Drawdown"
        ] = (
            merged["Close_Stock"]
            / rolling_max
            - 1
        )


        merged[
            "Max_Drawdown"
        ] = (
            merged["Drawdown"]
            .rolling(252)
            .min()
        )


        # ----------------------------------------------------
        # Beta
        # ----------------------------------------------------

        benchmark_returns = (
            merged["Close_Benchmark"]
            .pct_change()
        )


        merged[
            "Beta"
        ] = (
            stock_returns
            .rolling(60)
            .cov(
                benchmark_returns
            )
            /
            benchmark_returns
            .rolling(60)
            .var()
        )


        # ----------------------------------------------------
        # Sentiment
        # ----------------------------------------------------

        if not sentiment_df.empty:

            ticker_sentiment = (
                sentiment_df[
                    sentiment_df["ticker"]
                    == ticker
                ][
                    [
                        "Trading_Date",
                        "FinBERT_Sentiment_Mean",
                    ]
                ]
                .drop_duplicates(
                    "Trading_Date"
                )
            )


            merged = merged.merge(
                ticker_sentiment,
                left_on="Date",
                right_on="Trading_Date",
                how="left",
            )


            merged[
                "Sentiment"
            ] = (
                merged[
                    "FinBERT_Sentiment_Mean"
                ]
                .fillna(0.0)
            )

        else:

            merged[
                "Sentiment"
            ] = 0.0


        # ----------------------------------------------------
        # Remove unavailable observations
        # ----------------------------------------------------

        required_columns = [
            "Forecast",
            "Sentiment",
            "Volatility",
            "Max_Drawdown",
            "Beta",
            "Actual_Return",
        ]


        backtest = (
            merged[
                required_columns
                + ["Date"]
            ]
            .dropna()
            .copy()
        )


        if backtest.empty:

            print(
                f"Skipping {ticker}: "
                "insufficient historical data."
            )

            continue


        # ----------------------------------------------------
        # Use a historical period for evaluation
        # ----------------------------------------------------

        # Keep the most recent observations while
        # maintaining strictly historical inputs.

        backtest = (
            backtest
            .tail(LOOKBACK_DAYS)
            .copy()
        )


        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        recommendation_df = (
            generate_recommendations(
                forecast_returns=backtest[
                    "Forecast"
                ],
                sentiment_scores=backtest[
                    "Sentiment"
                ],
                volatility=backtest[
                    "Volatility"
                ],
                max_drawdown=backtest[
                    "Max_Drawdown"
                ],
                beta=backtest[
                    "Beta"
                ],
            )
        )


        backtest[
            "Recommendation"
        ] = (
            recommendation_df[
                "Recommendation"
            ]
            .values
        )


        # ----------------------------------------------------
        # Directional evaluation
        # ----------------------------------------------------

        directional = backtest[
            backtest[
                "Recommendation"
            ].isin(
                [
                    "BUY",
                    "SELL",
                ]
            )
        ].copy()


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


        # ----------------------------------------------------
        # BUY / SELL statistics
        # ----------------------------------------------------

        buy_count = (
            backtest[
                "Recommendation"
            ]
            .eq("BUY")
            .sum()
        )


        sell_count = (
            backtest[
                "Recommendation"
            ]
            .eq("SELL")
            .sum()
        )


        hold_count = (
            backtest[
                "Recommendation"
            ]
            .eq("HOLD")
            .sum()
        )


        # ----------------------------------------------------
        # Average realized return
        # ----------------------------------------------------

        directional_return = (
            directional[
                "Actual_Return"
            ]
            .mean()
            if not directional.empty
            else np.nan
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "Ticker": ticker,
                "Observations": len(
                    backtest
                ),
                "Directional_Calls": len(
                    directional
                ),
                "BUY_Count": int(
                    buy_count
                ),
                "HOLD_Count": int(
                    hold_count
                ),
                "SELL_Count": int(
                    sell_count
                ),
                "Hit_Rate_%": hit_rate,
                "Average_Directional_Return":
                    directional_return,
            }
        )


        print(
            f"{ticker}: "
            f"Hit Rate = "
            f"{hit_rate:.2f}%"
            if not pd.isna(hit_rate)
            else
            f"{ticker}: Hit Rate = N/A"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    if results_df.empty:

        raise RuntimeError(
            "No recommendation "
            "backtest results generated."
        )


    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )


    output_path = os.path.join(
        REPORTS_DIR,
        "recommendation_backtest.csv",
    )


    results_df.to_csv(
        output_path,
        index=False,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    valid_hit_rates = (
        results_df[
            "Hit_Rate_%"
        ]
        .dropna()
    )


    print()
    print("=" * 70)
    print("RECOMMENDATION BACKTEST SUMMARY")
    print("=" * 70)


    print(
        results_df.round(4)
    )


    if not valid_hit_rates.empty:

        print()
        print(
            "Average Directional Hit Rate: "
            f"{valid_hit_rates.mean():.2f}%"
        )


    print()
    print(
        f"Saved: {output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()