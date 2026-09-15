import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import prepare_modeling_data


SEED = 42
SEQ_LEN = 30
EPOCHS = 30
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def prepare_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def build_cnn_lstm(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))


def build_cnn_gru(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inputs)
    x = tf.keras.layers.GRU(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))


def build_attention_lstm(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=16
    )(x, x)

    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))


def build_mlp(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test, verbose=0).ravel()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    direction = (
        np.mean(np.sign(predictions) == np.sign(y_test)) * 100
    )

    return mae, rmse, r2, direction


def main():

    print("=" * 70)
    print("Northgate AI Stock Predictor")
    print("Extended Deep Learning Model Comparison")
    print("=" * 70)

    raw_dir = "data/processed"

    benchmark = pd.read_csv(
        f"{raw_dir}/GSPC.csv",
        parse_dates=["Date"]
    ).set_index("Date")

    vix = pd.read_csv(
        f"{raw_dir}/VIX.csv",
        parse_dates=["Date"]
    ).set_index("Date")

    tickers = [
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

    results = []

    for ticker in tickers:

        print("\n" + "-" * 70)
        print(f"Processing {ticker}")
        print("-" * 70)

        path = f"{raw_dir}/{ticker}.csv"

        try:
            stock = pd.read_csv(
                path,
                parse_dates=["Date"]
            ).set_index("Date")

            X_df, y, feature_df = prepare_modeling_data(
                stock,
                benchmark_df=benchmark,
                vix_df=vix,
            )

            X = X_df.values.astype(np.float32)
            y = y.values.astype(np.float32)

            split = int(len(X) * 0.8)

            X_train = X[:split]
            X_test = X[split:]

            y_train = y[:split]
            y_test = y[split:]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train_seq, y_train_seq = prepare_sequences(
                X_train,
                y_train,
                SEQ_LEN
            )

            X_test_seq, y_test_seq = prepare_sequences(
                X_test,
                y_test,
                SEQ_LEN
            )

            input_shape = X_train_seq.shape[1:]

            print(
                f"Train sequences: {X_train_seq.shape}"
            )
            print(
                f"Test sequences:  {X_test_seq.shape}"
            )

            models = {
                "CNN-LSTM": build_cnn_lstm(input_shape),
                "CNN-GRU": build_cnn_gru(input_shape),
                "Attention-LSTM": build_attention_lstm(input_shape),
                "MLP": build_mlp(input_shape),
            }

            for model_name, model in models.items():

                print(f"\nTraining {model_name}...")

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001
                    ),
                    loss="mse"
                )

                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )

                model.fit(
                    X_train_seq,
                    y_train_seq,
                    validation_split=0.1,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping],
                    verbose=0,
                    shuffle=False
                )

                mae, rmse, r2, direction = evaluate_model(
                    model,
                    X_test_seq,
                    y_test_seq
                )

                results.append({
                    "Ticker": ticker,
                    "Model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Directional_Accuracy": direction
                })

                print(
                    f"{model_name}: "
                    f"MAE={mae:.6f}, "
                    f"RMSE={rmse:.6f}, "
                    f"R2={r2:.6f}, "
                    f"Direction={direction:.2f}%"
                )

                del model
                tf.keras.backend.clear_session()

        except Exception as e:
            print(f"ERROR processing {ticker}: {e}")

    results_df = pd.DataFrame(results)

    os.makedirs("reports", exist_ok=True)

    output = "reports/dl_extended_model_comparison.csv"

    results_df.to_csv(output, index=False)

    print("\n" + "=" * 70)
    print("EXTENDED DL COMPARISON COMPLETED")
    print("=" * 70)

    print(results_df)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()