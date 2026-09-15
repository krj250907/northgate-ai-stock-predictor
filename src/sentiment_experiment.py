import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models_ml import (
    chronological_train_test_split,
    train_ridge_regression,
    predict_ridge_regression,
)


# ============================================================
# STOCKS WITH HISTORICAL NEWS COVERAGE
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
    "HD",
]


# ============================================================
# SENTIMENT FEATURES
# ============================================================

SENTIMENT_COLUMNS = [
    "FinBERT_Sentiment_Mean",
    "FinBERT_Sentiment_Std",
    "FinBERT_News_Volume",
    "Momentum_3D",
    "News_Volume",
    "News_Volume_Rolling_Mean",
    "News_Volume_Ratio",
    "News_Volume_Spike",
]


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================

def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy in percentage.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return (
        np.mean(
            np.sign(y_true) == np.sign(y_pred)
        )
        * 100
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred):
    """
    Evaluate regression predictions.
    """

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
        "Directional_Accuracy": directional_accuracy(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# REBUILD SENTIMENT FEATURES
# ============================================================

def rebuild_sentiment_features():
    """
    Rebuild sentiment features correctly.

    Important:
    - Momentum is calculated from each ticker's own
      complete price history.
    - Momentum uses only information available BEFORE
      the current trading session.
    - News volume rolling statistics are calculated
      separately for each ticker.
    - Current-day news volume is excluded from the
      rolling baseline using shift(1).
    """

    print()
    print("=" * 60)
    print("REBUILDING SENTIMENT FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Load daily FinBERT sentiment
    # --------------------------------------------------------

    sentiment_path = (
        "data/processed/finbert_daily_sentiment.csv"
    )

    daily_sentiment = pd.read_csv(
        sentiment_path
    )

    daily_sentiment["Trading_Date"] = pd.to_datetime(
        daily_sentiment["Trading_Date"]
    )

    print(
        f"Loaded daily sentiment rows: "
        f"{len(daily_sentiment)}"
    )

    all_features = []

    # --------------------------------------------------------
    # Process every ticker independently
    # --------------------------------------------------------

    for ticker in TICKERS:

        print()
        print(f"Processing: {ticker}")

        # ----------------------------------------------------
        # Sentiment for this ticker
        # ----------------------------------------------------

        sentiment = daily_sentiment[
            daily_sentiment["ticker"] == ticker
        ].copy()

        if sentiment.empty:
            print(
                f"No sentiment data found for {ticker}"
            )
            continue

        sentiment = sentiment.sort_values(
            "Trading_Date"
        )

        # ----------------------------------------------------
        # Load THIS ticker's price data
        # ----------------------------------------------------

        price_path = (
            f"data/processed/{ticker}.csv"
        )

        price = pd.read_csv(
            price_path,
            parse_dates=["Date"],
        )

        price = price[
            ["Date", "Close"]
        ].copy()

        price = price.sort_values(
            "Date"
        )

        # ----------------------------------------------------
        # Ticker-specific 3-day momentum
        #
        # pct_change(3) calculates the previous 3-day return.
        #
        # shift(1) is critical:
        # it prevents today's/current-session price
        # from being used to create today's feature.
        # ----------------------------------------------------

        price["Momentum_3D"] = (
            price["Close"]
            .pct_change(3)
            .shift(1)
        )

        momentum = price[
            [
                "Date",
                "Momentum_3D",
            ]
        ].rename(
            columns={
                "Date": "Trading_Date"
            }
        )

        # ----------------------------------------------------
        # Merge ticker-specific momentum with sentiment
        # ----------------------------------------------------

        ticker_sentiment = sentiment.merge(
            momentum,
            on="Trading_Date",
            how="left",
        )

        # ----------------------------------------------------
        # Add ticker explicitly
        # ----------------------------------------------------

        ticker_sentiment["ticker"] = ticker

        # ----------------------------------------------------
        # News volume
        #
        # FinBERT_News_Volume is the number of articles
        # available for that ticker on that trading day.
        # ----------------------------------------------------

        ticker_sentiment["News_Volume"] = (
            ticker_sentiment[
                "FinBERT_News_Volume"
            ]
        )

        # ----------------------------------------------------
        # Previous 20-day news-volume average
        #
        # shift(1) excludes the current day's news.
        # ----------------------------------------------------

        ticker_sentiment[
            "News_Volume_Rolling_Mean"
        ] = (
            ticker_sentiment[
                "News_Volume"
            ]
            .shift(1)
            .rolling(
                window=20,
                min_periods=5,
            )
            .mean()
        )

        # ----------------------------------------------------
        # News-volume ratio
        # ----------------------------------------------------

        ticker_sentiment[
            "News_Volume_Ratio"
        ] = (
            ticker_sentiment[
                "News_Volume"
            ]
            / ticker_sentiment[
                "News_Volume_Rolling_Mean"
            ]
        )

        # ----------------------------------------------------
        # News-volume spike
        #
        # Spike = current volume > 2x previous
        # 20-day average.
        # ----------------------------------------------------

        ticker_sentiment[
            "News_Volume_Spike"
        ] = (
            ticker_sentiment[
                "News_Volume_Ratio"
            ]
            > 2.0
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        all_features.append(
            ticker_sentiment
        )

        print(
            f"News days: {len(ticker_sentiment)}"
        )

        print(
            f"Momentum available: "
            f"{ticker_sentiment['Momentum_3D'].notna().sum()}"
        )

    # --------------------------------------------------------
    # Combine all tickers
    # --------------------------------------------------------

    combined = pd.concat(
        all_features,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "ticker",
            "Trading_Date",
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    output_path = (
        "data/processed/sentiment_features.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("SENTIMENT FEATURES REBUILT")
    print("=" * 60)

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Rows: {len(combined)}"
    )

    print(
        f"Columns: {len(combined.columns)}"
    )

    print(
        f"Tickers: {combined['ticker'].nunique()}"
    )

    print()
    print("Ticker distribution:")

    print(
        combined["ticker"].value_counts()
    )

    print()
    print("Missing values:")

    print(
        combined[
            SENTIMENT_COLUMNS
        ].isna().sum()
    )

    return combined


# ============================================================
# PREPARE DATA FOR ONE TICKER
# ============================================================

def prepare_ticker_data(ticker):

    from src.features import (
        prepare_modeling_data,
    )

    # --------------------------------------------------------
    # Load stock
    # --------------------------------------------------------

    stock = pd.read_csv(
        f"data/processed/{ticker}.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    benchmark = pd.read_csv(
        "data/processed/GSPC.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    # --------------------------------------------------------
    # Load VIX
    # --------------------------------------------------------

    vix = pd.read_csv(
        "data/processed/VIX.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    # --------------------------------------------------------
    # Build technical features
    # --------------------------------------------------------

    X, y, technical = prepare_modeling_data(
        stock,
        benchmark,
        vix,
    )

    # --------------------------------------------------------
    # Convert Date into merge column
    # --------------------------------------------------------

    technical = (
        technical
        .reset_index()
        .rename(
            columns={
                "Date": "Trading_Date"
            }
        )
    )

    # --------------------------------------------------------
    # Load corrected sentiment features
    # --------------------------------------------------------

    sentiment = pd.read_csv(
        "data/processed/sentiment_features.csv"
    )

    sentiment["Trading_Date"] = pd.to_datetime(
        sentiment["Trading_Date"]
    )

    # --------------------------------------------------------
    # Keep only this ticker
    # --------------------------------------------------------

    sentiment = sentiment[
        sentiment["ticker"] == ticker
    ].copy()

    # --------------------------------------------------------
    # Merge technical + sentiment
    #
    # Inner join is intentional.
    #
    # We do NOT replace missing news with neutral sentiment.
    # --------------------------------------------------------

    merged = technical.merge(
        sentiment,
        on="Trading_Date",
        how="inner",
    )

    # --------------------------------------------------------
    # Remove rows with incomplete sentiment features
    # --------------------------------------------------------

    merged = merged.dropna(
        subset=SENTIMENT_COLUMNS
    ).copy()

    # --------------------------------------------------------
    # Technical feature names
    # --------------------------------------------------------

    base_columns = list(
        X.columns
    )

    # --------------------------------------------------------
    # Dataset WITHOUT sentiment
    # --------------------------------------------------------

    X_base = merged[
        base_columns
    ].copy()

    # --------------------------------------------------------
    # Dataset WITH sentiment
    # --------------------------------------------------------

    X_sentiment = merged[
        base_columns
        + SENTIMENT_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_target = merged[
        "target_return"
    ].copy()

    return (
        X_base,
        X_sentiment,
        y_target,
        merged,
    )


# ============================================================
# RUN ONE SENTIMENT EXPERIMENT
# ============================================================

def run_sentiment_experiment(
    X_without_sentiment,
    X_with_sentiment,
    y,
    test_size=0.20,
):
    """
    Compare Ridge regression:
    1. Without sentiment
    2. With sentiment
    """

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        X_base_train,
        X_base_test,
        y_train,
        y_test,
    ) = chronological_train_test_split(
        X_without_sentiment,
        y,
        test_size=test_size,
    )

    (
        X_sent_train,
        X_sent_test,
        _,
        _,
    ) = chronological_train_test_split(
        X_with_sentiment,
        y,
        test_size=test_size,
    )

    # --------------------------------------------------------
    # Train base Ridge model
    # --------------------------------------------------------

    base_model = train_ridge_regression(
        X_base_train,
        y_train,
    )

    # --------------------------------------------------------
    # Train sentiment Ridge model
    # --------------------------------------------------------

    sentiment_model = train_ridge_regression(
        X_sent_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    base_predictions = (
        predict_ridge_regression(
            base_model,
            X_base_test,
        )
    )

    sentiment_predictions = (
        predict_ridge_regression(
            sentiment_model,
            X_sent_test,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    base_metrics = evaluate_model(
        y_test,
        base_predictions,
    )

    sentiment_metrics = evaluate_model(
        y_test,
        sentiment_predictions,
    )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "Model": "Ridge Without Sentiment",
                **base_metrics,
            },
            {
                "Model": "Ridge With Sentiment",
                **sentiment_metrics,
            },
        ]
    )

    # --------------------------------------------------------
    # Delta metrics
    #
    # Negative Delta_MAE = improvement
    #
    # Positive Delta_Directional_Accuracy = improvement
    # --------------------------------------------------------

    results["Delta_MAE"] = np.nan

    results[
        "Delta_Directional_Accuracy"
    ] = np.nan

    results.loc[
        results["Model"]
        == "Ridge With Sentiment",
        "Delta_MAE",
    ] = (
        sentiment_metrics["MAE"]
        - base_metrics["MAE"]
    )

    results.loc[
        results["Model"]
        == "Ridge With Sentiment",
        "Delta_Directional_Accuracy",
    ] = (
        sentiment_metrics[
            "Directional_Accuracy"
        ]
        - base_metrics[
            "Directional_Accuracy"
        ]
    )

    return results


# ============================================================
# RUN ALL TICKERS
# ============================================================

def run_all_tickers():

    # --------------------------------------------------------
    # FIRST: rebuild corrected sentiment features
    # --------------------------------------------------------

    rebuild_sentiment_features()

    # --------------------------------------------------------
    # Store all results
    # --------------------------------------------------------

    all_results = []

    # --------------------------------------------------------
    # Run each ticker
    # --------------------------------------------------------

    for ticker in TICKERS:

        print()
        print("=" * 60)
        print(
            f"RUNNING CORRECTED EXPERIMENT: {ticker}"
        )
        print("=" * 60)

        try:

            (
                X_base,
                X_sentiment,
                y,
                merged,
            ) = prepare_ticker_data(
                ticker
            )

            print(
                f"Usable rows: {len(merged)}"
            )

            print(
                "Date range: "
                f"{merged['Trading_Date'].min().date()} "
                f"to "
                f"{merged['Trading_Date'].max().date()}"
            )

            # ------------------------------------------------
            # Run experiment
            # ------------------------------------------------

            results = run_sentiment_experiment(
                X_base,
                X_sentiment,
                y,
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            for _, row in results.iterrows():

                all_results.append(
                    {
                        "Ticker": ticker,
                        "Model": row["Model"],
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "R2": row["R2"],
                        "Directional_Accuracy": row[
                            "Directional_Accuracy"
                        ],
                        "Delta_MAE": row[
                            "Delta_MAE"
                        ],
                        "Delta_Directional_Accuracy": row[
                            "Delta_Directional_Accuracy"
                        ],
                        "Usable_Rows": len(
                            merged
                        ),
                    }
                )

            print()
            print(
                results.to_string(
                    index=False
                )
            )

        except Exception as exc:

            print()
            print(
                f"ERROR for {ticker}: {exc}"
            )

    # --------------------------------------------------------
    # Final DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    output_path = (
        "reports/sentiment_impact_all_tickers.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL CORRECTED RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_all_tickers()