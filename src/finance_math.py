import numpy as np
import pandas as pd


def calculate_returns(prices):
    """
    Calculate simple percentage returns.
    """
    prices = pd.Series(prices).astype(float)
    return prices.pct_change().dropna()


def calculate_covariance_matrix(returns):
    """
    Calculate the sample covariance matrix of asset returns.
    """
    returns = pd.DataFrame(returns).dropna()
    return returns.cov()


def calculate_beta(asset_returns, benchmark_returns):
    """
    Calculate beta of an asset relative to a benchmark.

    Beta = Cov(asset, benchmark) / Var(benchmark)
    """
    asset_returns = pd.Series(asset_returns).astype(float)
    benchmark_returns = pd.Series(benchmark_returns).astype(float)

    combined = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    asset = combined.iloc[:, 0]
    benchmark = combined.iloc[:, 1]

    covariance = np.cov(asset, benchmark, ddof=1)[0, 1]
    benchmark_variance = np.var(benchmark, ddof=1)

    if benchmark_variance == 0:
        raise ValueError("Benchmark variance cannot be zero.")

    return covariance / benchmark_variance


def calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.0,
    periods_per_year=252,
):
    """
    Calculate annualized Sharpe ratio.

    Assumes returns are periodic returns and risk_free_rate
    is an annualized risk-free rate.
    """
    returns = pd.Series(returns).astype(float).dropna()

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    periodic_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year)
    ) - 1

    excess_returns = returns - periodic_risk_free_rate

    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        raise ValueError("Return volatility cannot be zero.")

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from a price series.
    """
    prices = pd.Series(prices).astype(float).dropna()

    running_max = prices.cummax()
    drawdowns = prices / running_max - 1

    return drawdowns.min()


def calculate_all_metrics(
    prices,
    benchmark_prices=None,
    risk_free_rate=0.0,
    periods_per_year=252,
):
    """
    Calculate the main financial metrics for an asset.
    """
    returns = calculate_returns(prices)

    results = {
        "Annualized_Return": (
            (1 + returns.mean()) ** periods_per_year - 1
        ),
        "Annualized_Volatility": (
            returns.std(ddof=1) * np.sqrt(periods_per_year)
        ),
        "Sharpe_Ratio": calculate_sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "Maximum_Drawdown": calculate_max_drawdown(prices),
    }

    if benchmark_prices is not None:
        benchmark_returns = calculate_returns(benchmark_prices)

        results["Beta"] = calculate_beta(
            returns,
            benchmark_returns,
        )

    return results