import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def chronological_train_test_split(X, y, test_size=0.20):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def naive_random_walk_prediction(X_test):
    if "Return_1D" not in X_test.columns:
        raise ValueError(
            "X_test must contain 'Return_1D' for the naive baseline."
        )

    return X_test["Return_1D"].copy()


def evaluate_predictions(y_true, y_pred):
    y_true = pd.Series(y_true).astype(float)

    y_pred = pd.Series(
        y_pred,
        index=y_true.index,
    ).astype(float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    non_zero = y_true != 0

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero]
                        - y_pred[non_zero]
                    )
                    / y_true[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    r2 = r2_score(
        y_true,
        y_pred,
    )

    directional_accuracy = (
        np.mean(
            np.sign(y_true)
            == np.sign(y_pred)
        )
        * 100
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
    }


def train_linear_regression(X_train, y_train):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_linear_regression(model, X_test):
    return model.predict(X_test)


def train_ridge_regression(
    X_train,
    y_train,
    alpha=1.0,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                Ridge(alpha=alpha),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_ridge_regression(model, X_test):
    return model.predict(X_test)


def train_random_forest(
    X_train,
    y_train,
):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_random_forest(model, X_test):
    return model.predict(X_test)


def train_xgboost(
    X_train,
    y_train,
):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_xgboost(model, X_test):
    return model.predict(X_test)


def train_svr(
    X_train,
    y_train,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(
                    C=1.0,
                    epsilon=0.01,
                    kernel="rbf",
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_svr(model, X_test):
    return model.predict(X_test)


def create_scaled_model(model):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def time_series_cross_validate(
    X,
    y,
    model,
    n_splits=5,
):
    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    fold_results = []

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_validation = X.iloc[validation_index]

        y_train = y.iloc[train_index]
        y_validation = y.iloc[validation_index]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = evaluate_predictions(
            y_validation,
            predictions,
        )

        metrics["Fold"] = fold
        metrics["Train_Size"] = len(
            train_index
        )
        metrics["Validation_Size"] = len(
            validation_index
        )

        fold_results.append(metrics)

    return pd.DataFrame(
        fold_results
    )


def summarize_cross_validation(
    results,
):
    metric_columns = [
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Directional_Accuracy",
    ]

    return pd.DataFrame(
        {
            "Mean": results[
                metric_columns
            ].mean(),
            "Std": results[
                metric_columns
            ].std(),
        }
    )


def compare_classical_models(
    X,
    y,
    n_splits=5,
):
    models = {
        "Linear Regression": create_scaled_model(
            LinearRegression()
        ),
        "Ridge Regression": create_scaled_model(
            Ridge(alpha=1.0)
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        ),
        "SVR": create_scaled_model(
            SVR(
                C=1.0,
                epsilon=0.01,
                kernel="rbf",
            )
        ),
    }

    summary_rows = []

    for model_name, model in models.items():
        print(
            f"\nRunning {model_name}..."
        )

        results = time_series_cross_validate(
            X,
            y,
            model,
            n_splits=n_splits,
        )

        summary = summarize_cross_validation(
            results
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Mean_RMSE": summary.loc[
                    "RMSE",
                    "Mean",
                ],
                "Mean_MAE": summary.loc[
                    "MAE",
                    "Mean",
                ],
                "Mean_MAPE": summary.loc[
                    "MAPE",
                    "Mean",
                ],
                "Mean_R2": summary.loc[
                    "R2",
                    "Mean",
                ],
                "Mean_Directional_Accuracy": summary.loc[
                    "Directional_Accuracy",
                    "Mean",
                ],
                "Std_MAE": summary.loc[
                    "MAE",
                    "Std",
                ],
                "Std_Directional_Accuracy": summary.loc[
                    "Directional_Accuracy",
                    "Std",
                ],
            }
        )

    comparison = pd.DataFrame(
        summary_rows
    )

    comparison = comparison.sort_values(
        by="Mean_MAE",
        ascending=True,
    ).reset_index(drop=True)

    return comparison