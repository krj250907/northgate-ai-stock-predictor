import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm, t


def calculate_jarque_bera(returns):
    """
    Calculate the Jarque-Bera normality test.
    """
    returns = pd.Series(returns).astype(float).dropna()

    statistic, p_value = jarque_bera(returns)

    return {
        "JB_Statistic": float(statistic),
        "P_Value": float(p_value),
    }


def calculate_t_var(returns, confidence_level=0.95, degrees_of_freedom=None):
    """
    Calculate one-period Value at Risk using a Student's t distribution.

    Returns VaR as a positive loss number.
    """
    returns = pd.Series(returns).astype(float).dropna()

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if degrees_of_freedom is None:
        degrees_of_freedom = len(returns) - 1

    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    mean_return = returns.mean()
    sample_std = returns.std(ddof=1)

    quantile = t.ppf(1 - confidence_level, df=degrees_of_freedom)

    return -(mean_return + quantile * sample_std)


def block_bootstrap_sharpe(
    returns,
    block_size=20,
    n_bootstrap=1000,
    risk_free_rate=0.0,
    periods_per_year=252,
    confidence_level=0.95,
    random_state=42,
):
    """
    Estimate a confidence interval for the annualized Sharpe ratio
    using moving block bootstrap resampling.
    """
    returns = pd.Series(returns).astype(float).dropna().to_numpy()

    if len(returns) < block_size:
        raise ValueError("block_size cannot exceed the number of returns.")

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    periodic_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year)
    ) - 1

    excess_returns = returns - periodic_risk_free_rate

    original_sharpe = (
        excess_returns.mean()
        / excess_returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    n = len(excess_returns)

    possible_starts = np.arange(n - block_size + 1)

    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        sampled_returns = []

        while len(sampled_returns) < n:
            start = rng.choice(possible_starts)
            block = excess_returns[start:start + block_size]
            sampled_returns.extend(block)

        sampled_returns = np.asarray(sampled_returns[:n])

        volatility = sampled_returns.std(ddof=1)

        if volatility == 0:
            continue

        sharpe = (
            sampled_returns.mean()
            / volatility
            * np.sqrt(periods_per_year)
        )

        bootstrap_sharpes.append(sharpe)

    bootstrap_sharpes = np.asarray(bootstrap_sharpes)

    alpha = 1 - confidence_level

    lower = np.quantile(
        bootstrap_sharpes,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_sharpes,
        1 - alpha / 2,
    )

    return {
        "Sharpe_Ratio": float(original_sharpe),
        "Confidence_Level": confidence_level,
        "CI_Lower": float(lower),
        "CI_Upper": float(upper),
        "Bootstrap_Samples": len(bootstrap_sharpes),
    }


def normality_summary(returns):
    """
    Provide a compact normality summary using Jarque-Bera.
    """
    returns = pd.Series(returns).astype(float).dropna()

    jb = calculate_jarque_bera(returns)

    return {
        "Mean": float(returns.mean()),
        "Std": float(returns.std(ddof=1)),
        "Skewness": float(returns.skew()),
        "Kurtosis": float(returns.kurt()),
        "JB_Statistic": jb["JB_Statistic"],
        "JB_P_Value": jb["P_Value"],
        "Normally_Distributed_At_5pct": jb["P_Value"] >= 0.05,
    }