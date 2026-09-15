import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


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

from features import (
    prepare_modeling_data,
)

from models_dl import (
    create_sequences,
    train_transformer_model,
    predict_transformer_model,
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

SEQUENCE_LENGTH = 30

TEST_SIZE = 0.20

EPOCHS = 30

BATCH_SIZE = 32


# ============================================================
# LOAD DATA
# ============================================================

def load_stock_data(ticker):

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

    return (
        stock,
        benchmark,
        vix,
    )


# ============================================================
# GENERATE HOLDOUT FORECASTS
# ============================================================

def generate_holdout_forecasts(ticker):

    print()
    print("=" * 70)
    print(
        f"TRANSFORMER HOLDOUT: {ticker}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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
        f"Modeling data: {X.shape}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    print(
        f"Test period: "
        f"{X_test.index[0].date()} "
        f"to "
        f"{X_test.index[-1].date()}"
    )

    # --------------------------------------------------------
    # Scale using TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        index=X_train.index,
        columns=X_train.columns,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        index=X_test.index,
        columns=X_test.columns,
    )

    # --------------------------------------------------------
    # Create training sequences
    # --------------------------------------------------------

    X_train_seq, y_train_seq = (
        create_sequences(
            X_train_scaled,
            y_train,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    # --------------------------------------------------------
    # Create TEST sequences
    #
    # IMPORTANT:
    # Test sequences are created only from test rows.
    # Therefore they do not cross the train/test boundary.
    # --------------------------------------------------------

    X_test_seq, y_test_seq = (
        create_sequences(
            X_test_scaled,
            y_test,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    print(
        f"Training sequences: "
        f"{X_train_seq.shape}"
    )

    print(
        f"Test sequences: "
        f"{X_test_seq.shape}"
    )

    # --------------------------------------------------------
    # Validation split from TRAINING sequences
    # --------------------------------------------------------

    validation_split = int(
        len(X_train_seq) * 0.90
    )

    X_fit = X_train_seq[
        :validation_split
    ]

    y_fit = y_train_seq[
        :validation_split
    ]

    X_validation = X_train_seq[
        validation_split:
    ]

    y_validation = y_train_seq[
        validation_split:
    ]

    print(
        f"Fit sequences: "
        f"{len(X_fit)}"
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
            X_fit,
            y_fit,
            X_validation,
            y_validation,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    )

    # --------------------------------------------------------
    # Predict TEST period
    # --------------------------------------------------------

    predictions = (
        predict_transformer_model(
            model,
            X_test_seq,
        )
    )

    # --------------------------------------------------------
    # Match sequence predictions to dates
    #
    # create_sequences starts prediction at index
    # SEQUENCE_LENGTH of the supplied test dataset.
    # --------------------------------------------------------

    prediction_dates = X_test.index[
        SEQUENCE_LENGTH:
    ]

    actual_returns = y_test.loc[
        prediction_dates
    ].values

    if len(predictions) != len(
        prediction_dates
    ):

        raise RuntimeError(
            "Prediction/date length mismatch."
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "Ticker": ticker,

            "Date": prediction_dates,

            "Predicted_Return":
                predictions,

            "Actual_Return":
                actual_returns,
        }
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    result[
        "Predicted_Direction"
    ] = np.where(
        result[
            "Predicted_Return"
        ] > 0,
        "UP",
        "DOWN",
    )

    result[
        "Actual_Direction"
    ] = np.where(
        result[
            "Actual_Return"
        ] > 0,
        "UP",
        "DOWN",
    )

    result[
        "Direction_Correct"
    ] = (
        result[
            "Predicted_Direction"
        ]
        ==
        result[
            "Actual_Direction"
        ]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            result[
                "Predicted_Return"
            ]
            -
            result[
                "Actual_Return"
            ]
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                result[
                    "Predicted_Return"
                ]
                -
                result[
                    "Actual_Return"
                ]
            )
            ** 2
        )
    )

    directional_accuracy = (
        result[
            "Direction_Correct"
        ]
        .mean()
        * 100
    )

    print()
    print(
        f"MAE: {mae:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"Directional Accuracy: "
        f"{directional_accuracy:.2f}%"
    )

    return (
        result,
        {
            "Ticker": ticker,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy":
                directional_accuracy,
            "Test_Observations":
                len(result),
        },
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )

    all_forecasts = []

    metrics = []

    print()
    print("=" * 70)
    print(
        "NORTHGATE HISTORICAL TRANSFORMER FORECASTS"
    )
    print("=" * 70)

    print(
        f"Stocks: {len(TICKERS)}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Test size: "
        f"{TEST_SIZE:.0%}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    # --------------------------------------------------------
    # Process all stocks
    # --------------------------------------------------------

    for ticker in TICKERS:

        try:

            forecast_df, metric = (
                generate_holdout_forecasts(
                    ticker
                )
            )

            all_forecasts.append(
                forecast_df
            )

            metrics.append(
                metric
            )

        except Exception as error:

            print()
            print(
                f"ERROR for {ticker}: "
                f"{error}"
            )

    if not all_forecasts:

        raise RuntimeError(
            "No historical forecasts "
            "were generated."
        )

    # --------------------------------------------------------
    # Combine forecasts
    # --------------------------------------------------------

    forecasts = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    forecasts = forecasts.sort_values(
        [
            "Ticker",
            "Date",
        ]
    ).reset_index(
        drop=True
    )

    forecasts_path = os.path.join(
        REPORTS_DIR,
        "transformer_holdout_forecasts.csv",
    )

    forecasts.to_csv(
        forecasts_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_path = os.path.join(
        REPORTS_DIR,
        "transformer_holdout_metrics.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRANSFORMER HOLDOUT SUMMARY"
    )
    print("=" * 70)

    print(
        metrics_df.round(6)
    )

    print()
    print(
        f"Saved forecasts: "
        f"{forecasts_path}"
    )

    print(
        f"Saved metrics: "
        f"{metrics_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()