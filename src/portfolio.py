import numpy as np
import pandas as pd


def calculate_expected_returns(
    returns,
    periods_per_year=252,
):
    """
    Calculate annualized arithmetic expected returns.
    """
    returns = pd.DataFrame(returns).dropna()

    return returns.mean() * periods_per_year


def calculate_covariance_matrix(
    returns,
    periods_per_year=252,
):
    """
    Calculate annualized covariance matrix.
    """
    returns = pd.DataFrame(returns).dropna()

    return returns.cov() * periods_per_year


def portfolio_return(weights, expected_returns):
    """
    Calculate portfolio expected annual return.
    """
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)

    return np.dot(weights, expected_returns)


def portfolio_volatility(weights, covariance_matrix):
    """
    Calculate annualized portfolio volatility.
    """
    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    variance = weights @ covariance_matrix @ weights

    return np.sqrt(max(variance, 0.0))


def portfolio_sharpe_ratio(
    weights,
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0,
):
    """
    Calculate portfolio Sharpe ratio.
    """
    expected_return = portfolio_return(
        weights,
        expected_returns,
    )

    volatility = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    if volatility == 0:
        raise ValueError("Portfolio volatility cannot be zero.")

    return (expected_return - risk_free_rate) / volatility


def generate_random_weights(
    n_assets,
    max_weight=0.30,
    random_state=None,
):
    """
    Generate long-only portfolio weights that sum to 1
    and respect the maximum weight constraint.
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")

    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1.")

    if n_assets * max_weight < 1:
        raise ValueError(
            "max_weight is too restrictive for the number of assets."
        )

    rng = np.random.default_rng(random_state)

    while True:
        weights = rng.dirichlet(np.ones(n_assets))

        if np.all(weights <= max_weight):
            return weights


def monte_carlo_portfolios(
    returns,
    n_portfolios=20000,
    max_weight=0.30,
    risk_free_rate=0.0,
    random_state=42,
):
    """
    Generate and evaluate random long-only portfolios.
    """
    returns = pd.DataFrame(returns).dropna()

    assets = returns.columns.tolist()

    expected_returns = calculate_expected_returns(returns)
    covariance_matrix = calculate_covariance_matrix(returns)

    results = []

    rng = np.random.default_rng(random_state)

    for _ in range(n_portfolios):
        weights = generate_random_weights(
            n_assets=len(assets),
            max_weight=max_weight,
            random_state=rng,
        )

        expected_return = portfolio_return(
            weights,
            expected_returns.values,
        )

        volatility = portfolio_volatility(
            weights,
            covariance_matrix.values,
        )

        sharpe = (
            (expected_return - risk_free_rate) / volatility
            if volatility > 0
            else np.nan
        )

        results.append(
            {
                "Return": expected_return,
                "Volatility": volatility,
                "Sharpe": sharpe,
                **{
                    asset: weight
                    for asset, weight in zip(assets, weights)
                },
            }
        )

    return pd.DataFrame(results)


def find_minimum_variance_portfolio(
    portfolio_results,
):
    """
    Return the minimum-variance portfolio from Monte Carlo results.
    """
    index = portfolio_results["Volatility"].idxmin()

    return portfolio_results.loc[index].copy()


def find_max_sharpe_portfolio(
    portfolio_results,
):
    """
    Return the maximum-Sharpe portfolio from Monte Carlo results.
    """
    index = portfolio_results["Sharpe"].idxmax()

    return portfolio_results.loc[index].copy()


def calculate_sortino_ratio(
    returns,
    target_return=0.0,
    periods_per_year=252,
):
    """
    Calculate annualized Sortino ratio.
    """
    returns = pd.Series(returns).astype(float).dropna()

    annualized_return = returns.mean() * periods_per_year

    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return np.inf

    downside_deviation = (
        np.sqrt(
            np.mean(
                (downside_returns - target_return) ** 2
            )
        )
        * np.sqrt(periods_per_year)
    )

    if downside_deviation == 0:
        return np.inf

    return annualized_return / downside_deviation


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from a price series.
    """
    prices = pd.Series(prices).astype(float).dropna()

    running_max = prices.cummax()

    drawdowns = prices / running_max - 1

    return drawdowns.min()


def calculate_portfolio_metrics(
    portfolio_returns,
    portfolio_prices=None,
    benchmark_returns=None,
    risk_free_rate=0.0,
):
    """
    Calculate portfolio risk and performance metrics.
    """
    portfolio_returns = pd.Series(
        portfolio_returns
    ).astype(float).dropna()

    annualized_return = (
        portfolio_returns.mean() * 252
    )

    annualized_volatility = (
        portfolio_returns.std(ddof=1)
        * np.sqrt(252)
    )

    sharpe = (
        (annualized_return - risk_free_rate)
        / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    sortino = calculate_sortino_ratio(
        portfolio_returns,
        target_return=risk_free_rate / 252,
    )

    metrics = {
        "Annualized_Return": annualized_return,
        "Annualized_Volatility": annualized_volatility,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
    }

    if portfolio_prices is not None:
        metrics["Maximum_Drawdown"] = calculate_max_drawdown(
            portfolio_prices
        )

    if benchmark_returns is not None:
        benchmark_returns = pd.Series(
            benchmark_returns
        ).astype(float)

        combined = pd.concat(
            [
                portfolio_returns.rename("Portfolio"),
                benchmark_returns.rename("Benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(combined) > 1:
            covariance = np.cov(
                combined["Portfolio"],
                combined["Benchmark"],
                ddof=1,
            )[0, 1]

            benchmark_variance = np.var(
                combined["Benchmark"],
                ddof=1,
            )

            if benchmark_variance > 0:
                metrics["Beta"] = (
                    covariance / benchmark_variance
                )

    return metrics