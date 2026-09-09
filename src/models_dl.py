import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_train_test_split(X, y, test_size=0.20):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def scale_train_test_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    return X_train_scaled, X_test_scaled, scaler


def create_sequences(X, y, sequence_length=30):
    X_values = np.asarray(X, dtype=np.float32)
    y_values = np.asarray(y, dtype=np.float32)

    if len(X_values) != len(y_values):
        raise ValueError("X and y must contain the same number of rows.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    if len(X_values) <= sequence_length:
        raise ValueError("Not enough observations to create sequences.")

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X_values)):
        X_sequences.append(
            X_values[i - sequence_length:i]
        )
        y_sequences.append(
            y_values[i]
        )

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_sequences, dtype=np.float32),
    )


def prepare_lstm_data(
    X,
    y,
    test_size=0.20,
    sequence_length=30,
):
    X_train, X_test, y_train, y_test = (
        chronological_train_test_split(
            X,
            y,
            test_size=test_size,
        )
    )

    X_train_scaled, X_test_scaled, scaler = (
        scale_train_test_data(
            X_train,
            X_test,
        )
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train,
        sequence_length=sequence_length,
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled,
        y_test,
        sequence_length=sequence_length,
    )

    return (
        X_train_seq,
        X_test_seq,
        y_train_seq,
        y_test_seq,
        scaler,
    )


# ============================================================
# LSTM
# ============================================================

def build_lstm_model(
    input_shape,
    units=64,
    dropout=0.2,
):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),

            tf.keras.layers.LSTM(
                units
            ),

            tf.keras.layers.Dropout(
                dropout
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


def train_lstm_model(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
    epochs=50,
    batch_size=32,
):
    import tensorflow as tf

    model = build_lstm_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    callbacks = []

    if (
        X_validation is not None
        and y_validation is not None
    ):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
        )

    return model, history


def predict_lstm_model(
    model,
    X_test,
):
    predictions = model.predict(
        X_test,
        verbose=0,
    )

    return predictions.reshape(-1)


# ============================================================
# GRU
# ============================================================

def build_gru_model(
    input_shape,
    units=64,
    dropout=0.2,
):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),

            tf.keras.layers.GRU(
                units
            ),

            tf.keras.layers.Dropout(
                dropout
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


def train_gru_model(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
    epochs=50,
    batch_size=32,
):
    import tensorflow as tf

    model = build_gru_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    callbacks = []

    if (
        X_validation is not None
        and y_validation is not None
    ):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
        )

    return model, history


def predict_gru_model(
    model,
    X_test,
):
    predictions = model.predict(
        X_test,
        verbose=0,
    )

    return predictions.reshape(-1)


# ============================================================
# BiLSTM
# ============================================================

def build_bilstm_model(
    input_shape,
    units=64,
    dropout=0.2,
):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units
                )
            ),

            tf.keras.layers.Dropout(
                dropout
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


def train_bilstm_model(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
    epochs=50,
    batch_size=32,
):
    import tensorflow as tf

    model = build_bilstm_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    callbacks = []

    if (
        X_validation is not None
        and y_validation is not None
    ):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
        )

    return model, history


def predict_bilstm_model(
    model,
    X_test,
):
    predictions = model.predict(
        X_test,
        verbose=0,
    )

    return predictions.reshape(-1)


# ============================================================
# TRANSFORMER
# ============================================================

def build_transformer_model(
    input_shape,
    num_heads=4,
    key_dim=16,
    ff_dim=64,
    dropout=0.2,
):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(
        shape=input_shape
    )

    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
    )(
        inputs,
        inputs,
    )

    attention_output = tf.keras.layers.Dropout(
        dropout
    )(attention_output)

    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(
        inputs + attention_output
    )

    feed_forward = tf.keras.layers.Dense(
        ff_dim,
        activation="relu",
    )(attention_output)

    feed_forward = tf.keras.layers.Dropout(
        dropout
    )(feed_forward)

    feed_forward = tf.keras.layers.Dense(
        input_shape[-1]
    )(feed_forward)

    encoder_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(
        attention_output + feed_forward
    )

    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(
        encoder_output
    )

    pooled_output = tf.keras.layers.Dropout(
        dropout
    )(pooled_output)

    dense_output = tf.keras.layers.Dense(
        32,
        activation="relu",
    )(pooled_output)

    outputs = tf.keras.layers.Dense(
        1
    )(dense_output)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


def train_transformer_model(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
    epochs=50,
    batch_size=32,
):
    import tensorflow as tf

    model = build_transformer_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    callbacks = []

    if (
        X_validation is not None
        and y_validation is not None
    ):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
        )

    return model, history


def predict_transformer_model(
    model,
    X_test,
):
    predictions = model.predict(
        X_test,
        verbose=0,
    )

    return predictions.reshape(-1)