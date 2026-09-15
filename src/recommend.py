import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_WEIGHT = 0.50
SENTIMENT_WEIGHT = 0.30
RISK_WEIGHT = 0.20

SELL_THRESHOLD = -0.20
BUY_THRESHOLD = 0.20

DEFAULT_REBALANCE_BAND = 0.05


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(series):
    """
    Normalize a pandas Series to the range [-1, 1].

    Higher values remain bullish.
    Lower values remain bearish.
    """

    series = pd.Series(series, dtype=float)

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            0.0,
            index=series.index,
        )

    if np.isclose(minimum, maximum):
        return pd.Series(
            0.0,
            index=series.index,
        )

    normalized = (
        2
        * (
            (series - minimum)
            / (maximum - minimum)
        )
        - 1
    )

    return normalized


# ============================================================
# FORECAST SIGNAL
# ============================================================

def calculate_forecast_signal(
    predicted_returns,
):
    """
    Convert predicted forward returns into
    normalized forecast signals.
    """

    predicted_returns = pd.Series(
        predicted_returns,
        dtype=float,
    )

    return min_max_normalize(
        predicted_returns
    )


# ============================================================
# SENTIMENT SIGNAL
# ============================================================

def calculate_sentiment_signal(
    sentiment_scores,
):
    """
    Convert sentiment scores into normalized
    bullish/bearish signals.
    """

    sentiment_scores = pd.Series(
        sentiment_scores,
        dtype=float,
    )

    return min_max_normalize(
        sentiment_scores
    )


# ============================================================
# RISK SIGNAL
# ============================================================

def calculate_risk_signal(
    volatility,
    max_drawdown,
    beta,
):
    """
    Create a normalized risk score.

    Higher volatility, larger drawdown and
    higher beta increase risk.

    The resulting score is normalized to [0, 1].
    """

    volatility = pd.Series(
        volatility,
        dtype=float,
    )

    max_drawdown = pd.Series(
        max_drawdown,
        dtype=float,
    )

    beta = pd.Series(
        beta,
        dtype=float,
    )

    # Drawdown is negative, therefore use absolute value.
    drawdown_risk = max_drawdown.abs()

    # Normalize each risk component.
    volatility_norm = min_max_normalize(
        volatility
    )

    drawdown_norm = min_max_normalize(
        drawdown_risk
    )

    beta_norm = min_max_normalize(
        beta
    )

    # Convert [-1, 1] to [0, 1].
    volatility_norm = (
        volatility_norm + 1
    ) / 2

    drawdown_norm = (
        drawdown_norm + 1
    ) / 2

    beta_norm = (
        beta_norm + 1
    ) / 2

    risk_score = (
        0.40 * volatility_norm
        + 0.30 * drawdown_norm
        + 0.30 * beta_norm
    )

    return risk_score


# ============================================================
# COMPOSITE SCORE
# ============================================================

def calculate_composite_score(
    forecast_signal,
    sentiment_signal,
    risk_signal,
    forecast_weight=FORECAST_WEIGHT,
    sentiment_weight=SENTIMENT_WEIGHT,
    risk_weight=RISK_WEIGHT,
):
    """
    Composite recommendation score.

    Composite =
        forecast_weight * forecast
        +
        sentiment_weight * sentiment
        -
        risk_weight * risk
    """

    score = (
        forecast_weight
        * forecast_signal
        +
        sentiment_weight
        * sentiment_signal
        -
        risk_weight
        * risk_signal
    )

    return score


# ============================================================
# RECOMMENDATION RULE
# ============================================================

def recommend(
    score,
    low=SELL_THRESHOLD,
    high=BUY_THRESHOLD,
):
    """
    Convert composite score into BUY / HOLD / SELL.
    """

    if score >= high:
        return "BUY"

    if score <= low:
        return "SELL"

    return "HOLD"


# ============================================================
# RECOMMENDATIONS FOR ALL ASSETS
# ============================================================

def generate_recommendations(
    forecast_returns,
    sentiment_scores,
    volatility,
    max_drawdown,
    beta,
):
    """
    Generate transparent BUY / HOLD / SELL
    recommendations for every asset.
    """

    forecast_returns = pd.Series(
        forecast_returns,
        dtype=float,
    )

    sentiment_scores = pd.Series(
        sentiment_scores,
        dtype=float,
    )

    volatility = pd.Series(
        volatility,
        dtype=float,
    )

    max_drawdown = pd.Series(
        max_drawdown,
        dtype=float,
    )

    beta = pd.Series(
        beta,
        dtype=float,
    )

    # --------------------------------------------------------
    # Align all assets
    # --------------------------------------------------------

    index = forecast_returns.index

    sentiment_scores = sentiment_scores.reindex(
        index
    )

    volatility = volatility.reindex(
        index
    )

    max_drawdown = max_drawdown.reindex(
        index
    )

    beta = beta.reindex(
        index
    )

    # --------------------------------------------------------
    # Calculate sub-signals
    # --------------------------------------------------------

    forecast_signal = (
        calculate_forecast_signal(
            forecast_returns
        )
    )

    sentiment_signal = (
        calculate_sentiment_signal(
            sentiment_scores
        )
    )

    risk_signal = calculate_risk_signal(
        volatility,
        max_drawdown,
        beta,
    )

    # --------------------------------------------------------
    # Composite
    # --------------------------------------------------------

    composite_score = (
        calculate_composite_score(
            forecast_signal,
            sentiment_signal,
            risk_signal,
        )
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = (
        composite_score.apply(
            recommend
        )
    )

    # --------------------------------------------------------
    # Output table
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "Predicted_Return":
                forecast_returns,

            "Sentiment":
                sentiment_scores,

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

            "Composite_Score":
                composite_score,

            "Recommendation":
                recommendation,
        }
    )

    return result


# ============================================================
# REBALANCING
# ============================================================

def rebalance(
    current_weights,
    target_weights,
    band=DEFAULT_REBALANCE_BAND,
):
    """
    Compare current and target portfolio weights.

    An action is generated only when the absolute
    drift exceeds the specified no-trade band.
    """

    current_weights = pd.Series(
        current_weights,
        dtype=float,
    )

    target_weights = pd.Series(
        target_weights,
        dtype=float,
    )

    assets = sorted(
        set(current_weights.index)
        | set(target_weights.index)
    )

    rows = []

    for asset in assets:

        current = current_weights.get(
            asset,
            0.0,
        )

        target = target_weights.get(
            asset,
            0.0,
        )

        drift = (
            target - current
        )

        if abs(drift) <= band:

            action = "HOLD"

        elif drift > 0:

            action = "INCREASE"

        else:

            action = "REDUCE"

        rows.append(
            {
                "Asset": asset,
                "Current_Weight": current,
                "Target_Weight": target,
                "Drift": drift,
                "Action": action,
            }
        )

    return pd.DataFrame(
        rows
    ).set_index("Asset")


# ============================================================
# RECOMMENDATION HIT RATE
# ============================================================

def calculate_recommendation_hit_rate(
    recommendations,
    actual_returns,
):
    """
    Evaluate historical recommendation direction.

    BUY  -> expected positive return
    SELL -> expected negative return
    HOLD -> not counted as directional prediction
    """

    recommendations = pd.Series(
        recommendations
    )

    actual_returns = pd.Series(
        actual_returns
    )

    aligned = pd.concat(
        [
            recommendations.rename(
                "Recommendation"
            ),
            actual_returns.rename(
                "Actual_Return"
            ),
        ],
        axis=1,
    ).dropna()

    directional = aligned[
        aligned["Recommendation"].isin(
            ["BUY", "SELL"]
        )
    ].copy()

    if directional.empty:
        return np.nan

    correct = (
        (
            (
                directional[
                    "Recommendation"
                ]
                == "BUY"
            )
            & (
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
            & (
                directional[
                    "Actual_Return"
                ]
                < 0
            )
        )
    )

    return (
        correct.mean()
        * 100
    )


# ============================================================
# TEST
# ============================================================

def _run_self_test():

    assets = [
        "AAPL",
        "MSFT",
        "JPM",
        "NVDA",
        "KO",
    ]

    forecast = pd.Series(
        {
            "AAPL": 0.012,
            "MSFT": 0.008,
            "JPM": -0.003,
            "NVDA": 0.020,
            "KO": -0.002,
        }
    )

    sentiment = pd.Series(
        {
            "AAPL": 0.50,
            "MSFT": 0.20,
            "JPM": -0.20,
            "NVDA": 0.70,
            "KO": -0.10,
        }
    )

    volatility = pd.Series(
        {
            "AAPL": 0.25,
            "MSFT": 0.22,
            "JPM": 0.18,
            "NVDA": 0.40,
            "KO": 0.15,
        }
    )

    drawdown = pd.Series(
        {
            "AAPL": -0.20,
            "MSFT": -0.18,
            "JPM": -0.15,
            "NVDA": -0.35,
            "KO": -0.10,
        }
    )

    beta = pd.Series(
        {
            "AAPL": 1.10,
            "MSFT": 1.05,
            "JPM": 1.20,
            "NVDA": 1.50,
            "KO": 0.60,
        }
    )

    result = generate_recommendations(
        forecast,
        sentiment,
        volatility,
        drawdown,
        beta,
    )

    assert len(result) == len(
        assets
    )

    assert set(
        result["Recommendation"]
    ).issubset(
        {
            "BUY",
            "HOLD",
            "SELL",
        }
    )

    print()
    print("=" * 60)
    print("RECOMMENDATION ENGINE TEST")
    print("=" * 60)

    print(
        result.round(4)
    )

    # --------------------------------------------------------
    # Rebalancing test
    # --------------------------------------------------------

    current_weights = pd.Series(
        {
            "AAPL": 0.10,
            "MSFT": 0.10,
            "JPM": 0.10,
            "NVDA": 0.10,
            "KO": 0.10,
        }
    )

    target_weights = pd.Series(
        {
            "AAPL": 0.20,
            "MSFT": 0.10,
            "JPM": 0.08,
            "NVDA": 0.25,
            "KO": 0.10,
        }
    )

    rebalance_table = rebalance(
        current_weights,
        target_weights,
        band=0.05,
    )

    assert (
        rebalance_table.loc[
            "AAPL",
            "Action",
        ]
        == "INCREASE"
    )

    assert (
        rebalance_table.loc[
            "MSFT",
            "Action",
        ]
        == "HOLD"
    )

    print()
    print("REBALANCING TEST")
    print(
        rebalance_table.round(4)
    )

    print()
    print("All recommendation tests passed.")


if __name__ == "__main__":
    _run_self_test()