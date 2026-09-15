import os
import sys

import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from features import prepare_modeling_data
from models_dl import (
    create_sequences,
    scale_train_test_data,
    train_transformer_model,
    predict_transformer_model,
)


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

SEQUENCE_LENGTH = 30

# Keep this moderate because Transformer training is expensive.
EPOCHS = 30
BATCH_SIZE = 32


# ============================================================
# LOAD DATA
# ============================================================

def load_stock_data(ticker):
    """
    Load processed stock, benchmark and VIX data.
    """

    stock_path = os.path.join(
        PROCESSED_DIR,
        f"{ticker}.csv",
    )

    benchmark_path = os.path.join(
        PROCESSED_DIR,
        "GSPC.csv",
    )

    vix_path = os.path.join(
        PROCESSED_DIR,
        "VIX.csv",
    )

    stock = pd.read_csv(
        stock_path,
        parse_dates=["Date"],
    )

    benchmark = pd.read_csv(
        benchmark_path,
        parse_dates=["Date"],
    )

    vix = pd.read_csv(
        vix_path,
        parse_dates=["Date"],
    )

    stock = (
        stock
        .set_index("Date")
        .sort_index()
    )

    benchmark = (
        benchmark
        .set_index("Date")
        .sort_index()
    )

    vix = (
        vix
        .set_index("Date")
        .sort_index()
    )

    return stock, benchmark, vix


# ============================================================
# CREATE FORECAST
# ============================================================

def generate_ticker_forecast(ticker):
    """
    Train the Transformer on the complete historical
    modeling dataset and generate a next-day forecast
    from the most recent available sequence.
    """

    print()
    print("=" * 70)
    print(f"GENERATING FORECAST: {ticker}")
    print("=" * 70)

    stock, benchmark, vix = (
        load_stock_data(ticker)
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    X, y, modeling_df = (
        prepare_modeling_data(
            stock,
            benchmark_df=benchmark,
            vix_df=vix,
        )
    )

    print(
        f"Modeling data: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    # We use all historical data to train the final
    # production model. No future observations exist
    # beyond the latest date.
    X_scaled = (
        X.copy()
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_scaled
    )

    X_scaled = pd.DataFrame(
        X_scaled,
        index=X.index,
        columns=X.columns,
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    X_sequences, y_sequences = (
        create_sequences(
            X_scaled,
            y,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    print(
        f"Sequence data: "
        f"{X_sequences.shape}"
    )

    # --------------------------------------------------------
    # Create validation split
    # --------------------------------------------------------

    split_index = int(
        len(X_sequences) * 0.90
    )

    X_train = X_sequences[
        :split_index
    ]

    y_train = y_sequences[
        :split_index
    ]

    X_validation = X_sequences[
        split_index:
    ]

    y_validation = y_sequences[
        split_index:
    ]

    print(
        f"Training sequences: "
        f"{len(X_train)}"
    )

    print(
        f"Validation sequences: "
        f"{len(X_validation)}"
    )

    # --------------------------------------------------------
    # Train Transformer
    # --------------------------------------------------------

    model, history = (
        train_transformer_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    )

    # --------------------------------------------------------
    # Most recent sequence
    # --------------------------------------------------------

    latest_sequence = (
        X_scaled
        .iloc[-SEQUENCE_LENGTH:]
        .values
    )

    latest_sequence = (
        latest_sequence
        .reshape(
            1,
            SEQUENCE_LENGTH,
            X_scaled.shape[1],
        )
    )

    # --------------------------------------------------------
    # Predict next-day return
    # --------------------------------------------------------

    prediction = (
        predict_transformer_model(
            model,
            latest_sequence,
        )
    )

    predicted_return = float(
        prediction[0]
    )

    latest_date = (
        modeling_df.index[-1]
    )

    print(
        f"Latest data date: "
        f"{latest_date.date()}"
    )

    print(
        f"Predicted next-day return: "
        f"{predicted_return:.6f}"
    )

    print(
        f"Predicted next-day return (%): "
        f"{predicted_return * 100:.4f}%"
    )

    return {
        "Ticker": ticker,
        "Latest_Date": latest_date,
        "Predicted_Return": predicted_return,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )

    results = []

    print(
        "=" * 70
    )

    print(
        "NORTHGATE TRANSFORMER FORECAST GENERATOR"
    )

    print(
        "=" * 70
    )

    print(
        f"Stocks: {len(TICKERS)}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    for ticker in TICKERS:

        try:

            result = (
                generate_ticker_forecast(
                    ticker
                )
            )

            results.append(
                result
            )

        except Exception as error:

            print()
            print(
                f"ERROR for {ticker}: "
                f"{error}"
            )

    if not results:
        raise RuntimeError(
            "No forecasts were generated."
        )

    forecasts = pd.DataFrame(
        results
    )

    forecasts = forecasts.sort_values(
        "Predicted_Return",
        ascending=False,
    )

    output_path = os.path.join(
        REPORTS_DIR,
        "model_forecasts.csv",
    )

    forecasts.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL FORECASTS"
    )

    print(
        "=" * 70
    )

    print(
        forecasts.round(6)
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()