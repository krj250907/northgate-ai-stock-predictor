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


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return (
        np.mean(
            np.sign(y_true) == np.sign(y_pred)
        )
        * 100
    )


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


def run_sentiment_experiment(
    X_without_sentiment,
    X_with_sentiment,
    y,
    test_size=0.20,
    alpha=1.0,
):
    """
    Compare Ridge regression with and without
    sentiment features using a chronological split.
    """

    (
        X_train_base,
        X_test_base,
        y_train,
        y_test,
    ) = chronological_train_test_split(
        X_without_sentiment,
        y,
        test_size=test_size,
    )

    (
        X_train_sentiment,
        X_test_sentiment,
        _,
        _,
    ) = chronological_train_test_split(
        X_with_sentiment,
        y,
        test_size=test_size,
    )

    base_model = train_ridge_regression(
        X_train_base,
        y_train,
        alpha=alpha,
    )

    sentiment_model = train_ridge_regression(
        X_train_sentiment,
        y_train,
        alpha=alpha,
    )

    base_predictions = predict_ridge_regression(
        base_model,
        X_test_base,
    )

    sentiment_predictions = predict_ridge_regression(
        sentiment_model,
        X_test_sentiment,
    )

    base_metrics = evaluate_model(
        y_test,
        base_predictions,
    )

    sentiment_metrics = evaluate_model(
        y_test,
        sentiment_predictions,
    )

    comparison = pd.DataFrame(
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

    comparison["Delta_MAE"] = np.nan
    comparison["Delta_Directional_Accuracy"] = np.nan

    comparison.loc[
        1,
        "Delta_MAE",
    ] = (
        comparison.loc[1, "MAE"]
        - comparison.loc[0, "MAE"]
    )

    comparison.loc[
        1,
        "Delta_Directional_Accuracy",
    ] = (
        comparison.loc[
            1,
            "Directional_Accuracy",
        ]
        - comparison.loc[
            0,
            "Directional_Accuracy",
        ]
    )

    return comparison