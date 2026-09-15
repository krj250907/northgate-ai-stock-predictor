import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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

BENCHMARK = "GSPC"

MAX_WEIGHT = 0.30

# Use 3 years of historical data at each optimization point.
TRAINING_YEARS = 3

# Rebalance every 3 months.
REBALANCE_MONTHS = 3

TRADING_DAYS = 252

RISK_FREE_RATE = 0.0


# ============================================================
# DATA LOADING
# ============================================================

def load_price_data():
    """
    Load adjusted close prices for all stocks and benchmark.
    """

    data = {}

    for ticker in TICKERS + [BENCHMARK]:

        path = os.path.join(
            PROCESSED_DIR,
            f"{ticker}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing processed file: {path}"
            )

        df = pd.read_csv(
            path,
            parse_dates=["Date"],
        )

        df = df.set_index("Date").sort_index()

        if "Close" not in df.columns:
            raise ValueError(
                f"'Close' column missing from {path}"
            )

        data[ticker] = df["Close"]

    prices = pd.concat(
        data,
        axis=1,
    ).dropna()

    return prices


# ============================================================
# RETURNS
# ============================================================

def calculate_returns(prices):
    """
    Calculate daily percentage returns.
    """

    return prices.pct_change().dropna()


# ============================================================
# PORTFOLIO MATHEMATICS
# ============================================================

def portfolio_return(weights, expected_returns):
    """
    Annualized expected portfolio return.
    """

    return np.dot(
        weights,
        expected_returns,
    )


def portfolio_volatility(weights, covariance):
    """
    Annualized portfolio volatility.
    """

    variance = np.dot(
        weights.T,
        np.dot(
            covariance,
            weights,
        ),
    )

    return np.sqrt(
        max(variance, 0.0)
    )


def portfolio_sharpe(
    weights,
    expected_returns,
    covariance,
):
    """
    Portfolio Sharpe ratio.
    """

    expected_return = portfolio_return(
        weights,
        expected_returns,
    )

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    if volatility <= 1e-12:
        return 0.0

    return (
        expected_return - RISK_FREE_RATE
    ) / volatility


# ============================================================
# WEIGHT CONSTRAINTS
# ============================================================

def validate_weights(weights):
    """
    Validate long-only weights with 30% maximum
    allocation per asset.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return (
        np.isclose(
            weights.sum(),
            1.0,
        )
        and np.all(
            weights >= -1e-10
        )
        and np.all(
            weights <= MAX_WEIGHT + 1e-10
        )
    )


# ============================================================
# MINIMUM VARIANCE
# ============================================================

def optimize_minimum_variance(
    expected_returns,
    covariance,
):
    """
    Find long-only minimum-variance portfolio
    with maximum 30% allocation per asset.
    """

    n_assets = len(
        expected_returns
    )

    initial_weights = np.ones(
        n_assets
    ) / n_assets

    objective = (
        lambda weights:
        portfolio_volatility(
            weights,
            covariance,
        )
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1.0,
        }
    ]

    bounds = [
        (0.0, MAX_WEIGHT)
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum variance optimization failed: "
            + result.message
        )

    weights = result.x

    if not validate_weights(
        weights
    ):
        raise RuntimeError(
            "Invalid minimum variance weights."
        )

    return weights


# ============================================================
# MAXIMUM SHARPE
# ============================================================

def optimize_max_sharpe(
    expected_returns,
    covariance,
):
    """
    Find maximum-Sharpe portfolio with
    long-only and maximum 30% asset allocation.
    """

    n_assets = len(
        expected_returns
    )

    initial_weights = np.ones(
        n_assets
    ) / n_assets

    objective = (
        lambda weights:
        -portfolio_sharpe(
            weights,
            expected_returns,
            covariance,
        )
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1.0,
        }
    ]

    bounds = [
        (0.0, MAX_WEIGHT)
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Maximum Sharpe optimization failed: "
            + result.message
        )

    weights = result.x

    if not validate_weights(
        weights
    ):
        raise RuntimeError(
            "Invalid maximum Sharpe weights."
        )

    return weights


# ============================================================
# EQUAL WEIGHT
# ============================================================

def equal_weight_portfolio(
    n_assets,
):
    """
    Equal-weight portfolio.
    """

    return np.ones(
        n_assets
    ) / n_assets


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    returns,
):
    """
    Calculate maximum drawdown from daily returns.
    """

    returns = pd.Series(
        returns
    ).dropna()

    wealth = (
        1.0 + returns
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_max
    ) - 1.0

    return drawdown.min()


# ============================================================
# SORTINO RATIO
# ============================================================

def calculate_sortino_ratio(
    returns,
    periods_per_year=TRADING_DAYS,
):
    """
    Annualized Sortino ratio.
    """

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    downside = returns[
        returns < 0
    ]

    if downside.empty:
        return np.inf

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside ** 2
            )
        )
        * np.sqrt(
            periods_per_year
        )
    )

    annualized_return = (
        returns.mean()
        * periods_per_year
    )

    if downside_deviation <= 1e-12:
        return np.nan

    return (
        annualized_return
        - RISK_FREE_RATE
    ) / downside_deviation


# ============================================================
# BETA
# ============================================================

def calculate_beta(
    portfolio_returns,
    benchmark_returns,
):
    """
    Calculate portfolio beta against benchmark.
    """

    aligned = pd.concat(
        [
            pd.Series(
                portfolio_returns,
                name="portfolio",
            ),
            pd.Series(
                benchmark_returns,
                name="benchmark",
            ),
        ],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    benchmark_variance = (
        aligned["benchmark"]
        .var(
            ddof=1
        )
    )

    if benchmark_variance <= 1e-12:
        return np.nan

    covariance = (
        aligned["portfolio"]
        .cov(
            aligned["benchmark"]
        )
    )

    return (
        covariance
        / benchmark_variance
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    returns,
    benchmark_returns=None,
):
    """
    Calculate portfolio performance metrics.
    """

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return {
            "Annualized_Return": np.nan,
            "Annualized_Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Beta": np.nan,
            "Max_Drawdown": np.nan,
        }

    annualized_return = (
        returns.mean()
        * TRADING_DAYS
    )

    annualized_volatility = (
        returns.std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS
        )
    )

    if annualized_volatility <= 1e-12:
        sharpe = np.nan
    else:
        sharpe = (
            annualized_return
            - RISK_FREE_RATE
        ) / annualized_volatility

    sortino = calculate_sortino_ratio(
        returns
    )

    if benchmark_returns is None:
        beta = np.nan
    else:
        beta = calculate_beta(
            returns,
            benchmark_returns,
        )

    max_drawdown = calculate_max_drawdown(
        returns
    )

    return {
        "Annualized_Return":
            annualized_return,

        "Annualized_Volatility":
            annualized_volatility,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Beta":
            beta,

        "Max_Drawdown":
            max_drawdown,
    }


# ============================================================
# TRAINING WINDOW
# ============================================================

def get_training_window(
    returns,
    rebalance_date,
):
    """
    Select only historical observations available
    before the rebalance date.

    No future information is used.
    """

    start_date = (
        rebalance_date
        - pd.DateOffset(
            years=TRAINING_YEARS
        )
    )

    training = returns.loc[
        (returns.index >= start_date)
        & (
            returns.index
            < rebalance_date
        )
    ]

    return training


# ============================================================
# OPTIMIZE PORTFOLIO
# ============================================================

def optimize_portfolios(
    training_returns,
):
    """
    Calculate equal-weight, minimum-variance,
    and maximum-Sharpe weights.
    """

    assets = list(
        training_returns.columns
    )

    n_assets = len(
        assets
    )

    expected_returns = (
        training_returns.mean()
        .values
        * TRADING_DAYS
    )

    covariance = (
        training_returns.cov()
        .values
        * TRADING_DAYS
    )

    equal_weights = (
        equal_weight_portfolio(
            n_assets
        )
    )

    minimum_variance_weights = (
        optimize_minimum_variance(
            expected_returns,
            covariance,
        )
    )

    maximum_sharpe_weights = (
        optimize_max_sharpe(
            expected_returns,
            covariance,
        )
    )

    return {
        "Equal Weight":
            equal_weights,

        "Minimum Variance":
            minimum_variance_weights,

        "Maximum Sharpe":
            maximum_sharpe_weights,
    }


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def run_walkforward_backtest(
    prices,
):
    """
    Walk-forward portfolio backtest.

    At each quarterly rebalance:

    1. Use only the previous 3 years of returns.
    2. Optimize portfolio weights.
    3. Apply those weights to the next quarter.
    4. Move forward.
    5. Repeat.

    This prevents future information from entering
    portfolio construction.
    """

    returns = calculate_returns(
        prices
    )

    stock_returns = returns[
        TICKERS
    ]

    benchmark_returns = returns[
        BENCHMARK
    ]

    first_date = stock_returns.index.min()

    first_rebalance = (
        first_date
        + pd.DateOffset(
            years=TRAINING_YEARS
        )
    )

    # Move to the first available trading date.
    available_dates = stock_returns.index[
        stock_returns.index
        >= first_rebalance
    ]

    if len(available_dates) == 0:
        raise RuntimeError(
            "Not enough historical data "
            "for walk-forward backtest."
        )

    rebalance_date = (
        available_dates[0]
    )

    portfolio_names = [
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe",
    ]

    portfolio_returns = {
        name: []
        for name in portfolio_names
    }

    portfolio_dates = []

    weight_history = []

    while (
        rebalance_date
        < stock_returns.index.max()
    ):

        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        training_returns = (
            get_training_window(
                stock_returns,
                rebalance_date,
            )
        )

        if len(training_returns) < 252:
            break

        # ----------------------------------------------------
        # Optimize using historical data only
        # ----------------------------------------------------

        weights = optimize_portfolios(
            training_returns
        )

        # ----------------------------------------------------
        # Determine next rebalance date
        # ----------------------------------------------------

        next_rebalance = (
            rebalance_date
            + pd.DateOffset(
                months=REBALANCE_MONTHS
            )
        )

        test_period = stock_returns.loc[
            (
                stock_returns.index
                >= rebalance_date
            )
            & (
                stock_returns.index
                < next_rebalance
            )
        ]

        if test_period.empty:
            break

        # ----------------------------------------------------
        # Save weights
        # ----------------------------------------------------

        for portfolio_name, weight_array in weights.items():

            row = {
                "Rebalance_Date":
                    rebalance_date,

                "Portfolio":
                    portfolio_name,
            }

            for ticker, weight in zip(
                TICKERS,
                weight_array,
            ):
                row[ticker] = weight

            weight_history.append(
                row
            )

        # ----------------------------------------------------
        # Apply weights to future period
        # ----------------------------------------------------

        for portfolio_name, weight_array in weights.items():

            daily_returns = (
                test_period
                .values
                @ weight_array
            )

            portfolio_returns[
                portfolio_name
            ].extend(
                daily_returns.tolist()
            )

        portfolio_dates.extend(
            test_period.index.tolist()
        )

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        future_dates = stock_returns.index[
            stock_returns.index
            >= next_rebalance
        ]

        if len(future_dates) == 0:
            break

        rebalance_date = (
            future_dates[0]
        )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    performance = pd.DataFrame(
        portfolio_returns,
        index=pd.Index(
            portfolio_dates,
            name="Date",
        ),
    )

    # Remove any duplicate dates.
    performance = (
        performance[
            ~performance.index.duplicated(
                keep="first"
            )
        ]
    )

    benchmark = (
        benchmark_returns
        .reindex(
            performance.index
        )
    )

    performance[
        "S&P 500 Benchmark"
    ] = benchmark.values

    weights_df = pd.DataFrame(
        weight_history
    )

    return (
        performance,
        weights_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )

    print(
        "=" * 70
    )
    print(
        "NORTHGATE WALK-FORWARD PORTFOLIO BACKTEST"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load prices
    # --------------------------------------------------------

    prices = load_price_data()

    print()
    print(
        f"Price observations: {len(prices)}"
    )

    print(
        f"Date range: "
        f"{prices.index.min().date()} "
        f"to "
        f"{prices.index.max().date()}"
    )

    # --------------------------------------------------------
    # Run walk-forward
    # --------------------------------------------------------

    performance, weights = (
        run_walkforward_backtest(
            prices
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "WALK-FORWARD CONFIGURATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Training window: "
        f"{TRAINING_YEARS} years"
    )

    print(
        f"Rebalance frequency: "
        f"Every {REBALANCE_MONTHS} months"
    )

    print(
        f"Maximum asset weight: "
        f"{MAX_WEIGHT:.0%}"
    )

    print(
        f"Backtest observations: "
        f"{len(performance)}"
    )

    print(
        f"Backtest period: "
        f"{performance.index.min().date()} "
        f"to "
        f"{performance.index.max().date()}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    benchmark = performance[
        "S&P 500 Benchmark"
    ]

    metrics = {}

    for portfolio in [
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe",
        "S&P 500 Benchmark",
    ]:

        portfolio_returns = (
            performance[
                portfolio
            ]
        )

        benchmark_for_beta = (
            None
            if portfolio
            == "S&P 500 Benchmark"
            else benchmark
        )

        metrics[
            portfolio
        ] = calculate_metrics(
            portfolio_returns,
            benchmark_for_beta,
        )

    metrics_df = pd.DataFrame(
        metrics
    ).T

    # Benchmark beta is exactly 1 by definition.
    metrics_df.loc[
        "S&P 500 Benchmark",
        "Beta",
    ] = 1.0

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE"
    )
    print(
        "=" * 70
    )

    print(
        metrics_df.round(4)
    )

    # --------------------------------------------------------
    # Save performance
    # --------------------------------------------------------

    performance_path = os.path.join(
        REPORTS_DIR,
        "portfolio_walkforward.csv",
    )

    performance.to_csv(
        performance_path
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        REPORTS_DIR,
        "portfolio_walkforward_metrics.csv",
    )

    metrics_df.to_csv(
        metrics_path
    )

    # --------------------------------------------------------
    # Save weights
    # --------------------------------------------------------

    weights_path = os.path.join(
        REPORTS_DIR,
        "portfolio_walkforward_weights.csv",
    )

    weights.to_csv(
        weights_path,
        index=False,
    )

    # --------------------------------------------------------
    # Acceptance comparison
    # --------------------------------------------------------

    equal_sharpe = metrics_df.loc[
        "Equal Weight",
        "Sharpe",
    ]

    max_sharpe = metrics_df.loc[
        "Maximum Sharpe",
        "Sharpe",
    ]

    minimum_variance_sharpe = (
        metrics_df.loc[
            "Minimum Variance",
            "Sharpe",
        ]
    )

    print()
    print(
        "=" * 70
    )
    print(
        "WALK-FORWARD COMPARISON"
    )
    print(
        "=" * 70
    )

    print(
        f"Equal Weight Sharpe: "
        f"{equal_sharpe:.4f}"
    )

    print(
        f"Minimum Variance Sharpe: "
        f"{minimum_variance_sharpe:.4f}"
    )

    print(
        f"Maximum Sharpe Sharpe: "
        f"{max_sharpe:.4f}"
    )

    if max_sharpe > equal_sharpe:
        print(
            "PASS: Maximum-Sharpe portfolio "
            "beats Equal Weight on Sharpe."
        )
    else:
        print(
            "NOTE: Maximum-Sharpe portfolio "
            "does not beat Equal Weight on Sharpe."
        )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {performance_path}"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {weights_path}"
    )


if __name__ == "__main__":
    main()