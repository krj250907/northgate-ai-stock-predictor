import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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

BENCHMARK = "GSPC"

TRAIN_END = "2024-12-31"

MAX_WEIGHT = 0.30

TRADING_DAYS = 252

RISK_FREE_RATE = 0.0


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data():
    """
    Load adjusted/cleaned closing prices for all assets.
    """

    price_data = {}

    for ticker in TICKERS + [BENCHMARK]:

        path = f"data/processed/{ticker}.csv"

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_csv(
            path,
            parse_dates=["Date"],
        )

        df = df[
            ["Date", "Close"]
        ].copy()

        df = df.sort_values("Date")

        df = df.drop_duplicates(
            subset="Date"
        )

        df = df.set_index("Date")

        price_data[ticker] = df[
            "Close"
        ]

    prices = pd.concat(
        price_data,
        axis=1,
        join="inner",
    )

    prices.columns = (
        TICKERS + [BENCHMARK]
    )

    prices = prices.dropna()

    return prices


# ============================================================
# CALCULATE RETURNS
# ============================================================

def calculate_returns(prices):
    """
    Calculate daily simple returns.
    """

    returns = prices.pct_change()

    returns = returns.dropna()

    return returns


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def portfolio_return(
    weights,
    expected_returns,
):
    """
    Annualized expected portfolio return.
    """

    return np.dot(
        weights,
        expected_returns,
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def portfolio_volatility(
    weights,
    covariance_matrix,
):
    """
    Annualized portfolio volatility.
    """

    variance = np.dot(
        weights.T,
        np.dot(
            covariance_matrix,
            weights,
        ),
    )

    return np.sqrt(
        max(variance, 0)
    )


# ============================================================
# PORTFOLIO SHARPE
# ============================================================

def portfolio_sharpe(
    weights,
    expected_returns,
    covariance_matrix,
):
    """
    Annualized Sharpe ratio.
    """

    ret = portfolio_return(
        weights,
        expected_returns,
    )

    vol = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    if vol == 0:
        return 0.0

    return (
        ret - RISK_FREE_RATE
    ) / vol


# ============================================================
# OPTIMIZATION CONSTRAINTS
# ============================================================

def optimization_constraints():
    """
    Portfolio weights must sum to 1.
    """

    return {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1.0,
    }


# ============================================================
# MINIMUM VARIANCE
# ============================================================

def find_minimum_variance(
    expected_returns,
    covariance_matrix,
):
    """
    Find long-only minimum-variance portfolio
    with maximum 30% weight per asset.
    """

    n_assets = len(
        expected_returns
    )

    initial_weights = np.ones(
        n_assets
    ) / n_assets

    bounds = [
        (0.0, MAX_WEIGHT)
        for _ in range(n_assets)
    ]

    result = minimize(
        lambda weights: portfolio_volatility(
            weights,
            covariance_matrix,
        ),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=optimization_constraints(),
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum variance optimization failed: "
            + result.message
        )

    return result.x


# ============================================================
# MAXIMUM SHARPE
# ============================================================

def find_max_sharpe(
    expected_returns,
    covariance_matrix,
):
    """
    Find maximum-Sharpe portfolio
    with long-only 30% asset cap.
    """

    n_assets = len(
        expected_returns
    )

    initial_weights = np.ones(
        n_assets
    ) / n_assets

    bounds = [
        (0.0, MAX_WEIGHT)
        for _ in range(n_assets)
    ]

    result = minimize(
        lambda weights: -portfolio_sharpe(
            weights,
            expected_returns,
            covariance_matrix,
        ),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=optimization_constraints(),
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Maximum Sharpe optimization failed: "
            + result.message
        )

    return result.x


# ============================================================
# PORTFOLIO DAILY RETURNS
# ============================================================

def calculate_portfolio_returns(
    returns,
    weights,
):
    """
    Calculate daily portfolio returns.
    """

    asset_returns = returns[
        TICKERS
    ]

    portfolio_returns = (
        asset_returns
        @ weights
    )

    return portfolio_returns


# ============================================================
# ANNUALIZED RETURN
# ============================================================

def annualized_return(
    daily_returns,
):
    """
    Annualized compounded return.
    """

    daily_returns = (
        daily_returns
        .dropna()
    )

    if len(daily_returns) == 0:
        return np.nan

    total_return = (
        np.prod(
            1 + daily_returns
        )
        ** (
            TRADING_DAYS
            / len(daily_returns)
        )
        - 1
    )

    return total_return


# ============================================================
# ANNUALIZED VOLATILITY
# ============================================================

def annualized_volatility(
    daily_returns,
):
    """
    Annualized volatility.
    """

    return (
        daily_returns.std()
        * np.sqrt(TRADING_DAYS)
    )


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe(
    daily_returns,
):
    """
    Annualized Sharpe ratio.
    """

    vol = annualized_volatility(
        daily_returns
    )

    if vol == 0:
        return np.nan

    return (
        annualized_return(
            daily_returns
        )
        - RISK_FREE_RATE
    ) / vol


# ============================================================
# SORTINO RATIO
# ============================================================

def calculate_sortino(
    daily_returns,
):
    """
    Annualized Sortino ratio using 0% daily target.
    """

    downside = np.minimum(
        daily_returns,
        0,
    )

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside ** 2
            )
        )
        * np.sqrt(TRADING_DAYS)
    )

    if downside_deviation == 0:
        return np.nan

    return (
        annualized_return(
            daily_returns
        )
        - RISK_FREE_RATE
    ) / downside_deviation


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    daily_returns,
):
    """
    Calculate maximum drawdown.
    """

    wealth = (
        1 + daily_returns
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_max
    ) - 1

    return drawdown.min()


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
            portfolio_returns,
            benchmark_returns,
        ],
        axis=1,
    ).dropna()

    portfolio = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    covariance = np.cov(
        portfolio,
        benchmark,
        ddof=1,
    )[0, 1]

    benchmark_variance = np.var(
        benchmark,
        ddof=1,
    )

    if benchmark_variance == 0:
        return np.nan

    return (
        covariance
        / benchmark_variance
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    daily_returns,
    benchmark_returns,
):
    """
    Calculate complete portfolio performance metrics.
    """

    return {
        "Annualized_Return":
            annualized_return(
                daily_returns
            ),

        "Annualized_Volatility":
            annualized_volatility(
                daily_returns
            ),

        "Sharpe":
            calculate_sharpe(
                daily_returns
            ),

        "Sortino":
            calculate_sortino(
                daily_returns
            ),

        "Beta":
            calculate_beta(
                daily_returns,
                benchmark_returns,
            ),

        "Max_Drawdown":
            calculate_max_drawdown(
                daily_returns
            ),
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest():

    print()
    print("=" * 70)
    print("NORTHGATE PORTFOLIO OUT-OF-SAMPLE BACKTEST")
    print("=" * 70)

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
    # Daily returns
    # --------------------------------------------------------

    returns = calculate_returns(
        prices
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    train_returns = returns[
        returns.index <= TRAIN_END
    ].copy()

    test_returns = returns[
        returns.index > TRAIN_END
    ].copy()

    if train_returns.empty:
        raise ValueError(
            "Training period is empty."
        )

    if test_returns.empty:
        raise ValueError(
            "Test period is empty."
        )

    print()
    print(
        "TRAINING / OPTIMIZATION PERIOD"
    )

    print(
        f"{train_returns.index.min().date()} "
        f"to "
        f"{train_returns.index.max().date()}"
    )

    print(
        f"Training observations: "
        f"{len(train_returns)}"
    )

    print()
    print(
        "OUT-OF-SAMPLE TEST PERIOD"
    )

    print(
        f"{test_returns.index.min().date()} "
        f"to "
        f"{test_returns.index.max().date()}"
    )

    print(
        f"Test observations: "
        f"{len(test_returns)}"
    )

    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    asset_train_returns = (
        train_returns[
            TICKERS
        ]
    )

    expected_returns = (
        asset_train_returns
        .mean()
        .values
        * TRADING_DAYS
    )

    covariance_matrix = (
        asset_train_returns
        .cov()
        .values
        * TRADING_DAYS
    )

    # --------------------------------------------------------
    # Equal-weight portfolio
    # --------------------------------------------------------

    equal_weights = (
        np.ones(
            len(TICKERS)
        )
        / len(TICKERS)
    )

    # --------------------------------------------------------
    # Minimum variance portfolio
    # --------------------------------------------------------

    min_variance_weights = (
        find_minimum_variance(
            expected_returns,
            covariance_matrix,
        )
    )

    # --------------------------------------------------------
    # Maximum Sharpe portfolio
    # --------------------------------------------------------

    max_sharpe_weights = (
        find_max_sharpe(
            expected_returns,
            covariance_matrix,
        )
    )

    # --------------------------------------------------------
    # Verify weights
    # --------------------------------------------------------

    portfolios = {
        "Equal Weight": equal_weights,
        "Minimum Variance": min_variance_weights,
        "Maximum Sharpe": max_sharpe_weights,
    }

    print()
    print("=" * 70)
    print("OPTIMIZED WEIGHTS")
    print("=" * 70)

    weights_table = pd.DataFrame(
        portfolios,
        index=TICKERS,
    )

    weights_table["Equal Weight"] *= 100
    weights_table["Minimum Variance"] *= 100
    weights_table["Maximum Sharpe"] *= 100

    print(
        weights_table.round(2)
    )

    # --------------------------------------------------------
    # Weight validation
    # --------------------------------------------------------

    for name, weights in portfolios.items():

        if not np.isclose(
            weights.sum(),
            1.0,
        ):
            raise ValueError(
                f"{name} weights do not sum to 1."
            )

        if np.max(weights) > (
            MAX_WEIGHT + 1e-8
        ):
            raise ValueError(
                f"{name} violates the "
                f"{MAX_WEIGHT:.0%} asset cap."
            )

    # --------------------------------------------------------
    # Out-of-sample returns
    # --------------------------------------------------------

    benchmark_test_returns = (
        test_returns[
            BENCHMARK
        ]
    )

    performance = {}

    # --------------------------------------------------------
    # Equal Weight
    # --------------------------------------------------------

    performance[
        "Equal Weight"
    ] = calculate_metrics(
        calculate_portfolio_returns(
            test_returns,
            equal_weights,
        ),
        benchmark_test_returns,
    )

    # --------------------------------------------------------
    # Minimum Variance
    # --------------------------------------------------------

    performance[
        "Minimum Variance"
    ] = calculate_metrics(
        calculate_portfolio_returns(
            test_returns,
            min_variance_weights,
        ),
        benchmark_test_returns,
    )

    # --------------------------------------------------------
    # Maximum Sharpe
    # --------------------------------------------------------

    performance[
        "Maximum Sharpe"
    ] = calculate_metrics(
        calculate_portfolio_returns(
            test_returns,
            max_sharpe_weights,
        ),
        benchmark_test_returns,
    )

    # --------------------------------------------------------
    # S&P 500 benchmark
    # --------------------------------------------------------

    performance[
        "S&P 500 Benchmark"
    ] = calculate_metrics(
        benchmark_test_returns,
        benchmark_test_returns,
    )

    # --------------------------------------------------------
    # Performance DataFrame
    # --------------------------------------------------------

    performance_df = (
        pd.DataFrame(
            performance
        ).T
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    weights_path = (
        "reports/portfolio_weights.csv"
    )

    performance_path = (
        "reports/portfolio_backtest.csv"
    )

    weights_table.to_csv(
        weights_path
    )

    performance_df.to_csv(
        performance_path
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 70)

    print(
        performance_df.round(4)
    )

    print()
    print(
        f"Saved weights: {weights_path}"
    )

    print(
        f"Saved backtest: {performance_path}"
    )

    # --------------------------------------------------------
    # Acceptance check
    # --------------------------------------------------------

    max_sharpe_value = (
        performance_df.loc[
            "Maximum Sharpe",
            "Sharpe",
        ]
    )

    equal_weight_value = (
        performance_df.loc[
            "Equal Weight",
            "Sharpe",
        ]
    )

    print()
    print("=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)

    if (
        max_sharpe_value
        > equal_weight_value
    ):
        print(
            "PASS: Maximum-Sharpe portfolio "
            "has higher out-of-sample Sharpe "
            "than Equal Weight."
        )
    else:
        print(
            "NOTE: Maximum-Sharpe portfolio "
            "does NOT beat Equal Weight "
            "on out-of-sample Sharpe."
        )

    return (
        weights_table,
        performance_df,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_backtest()