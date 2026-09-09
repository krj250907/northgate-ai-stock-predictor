import pandas as pd


def add_return_features(df):
    df = df.copy()

    df["Return_1D"] = df["Close"].pct_change()
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Return_20D"] = df["Close"].pct_change(20)

    return df


def add_moving_average_features(df):
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    return df


def add_macd_features(df):
    df = df.copy()

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    return df


def add_rsi_features(df, window=14):
    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


def add_stochastic_features(df, window=14):
    df = df.copy()

    lowest_low = df["Low"].rolling(window=window).min()
    highest_high = df["High"].rolling(window=window).max()

    df["Stoch_K"] = (
        100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    )

    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    return df


def add_volatility_features(df):
    df = df.copy()

    df["Volatility_20D"] = df["Return_1D"].rolling(window=20).std()

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR_14"] = true_range.rolling(window=14).mean()

    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()

    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)

    df["Bollinger_Width"] = (upper_band - lower_band) / sma_20

    return df


def add_volume_features(df, window=20):
    df = df.copy()

    volume_mean = df["Volume"].rolling(window=window).mean()
    volume_std = df["Volume"].rolling(window=window).std()

    df["Volume_ZScore"] = (
        (df["Volume"] - volume_mean) / volume_std
    )

    price_direction = df["Close"].diff()

    direction = price_direction.apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    df["OBV"] = (direction * df["Volume"]).cumsum()

    return df


def add_calendar_features(df):
    df = df.copy()

    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    return df


def add_lag_features(df):
    df = df.copy()

    df["Return_1D_Lag1"] = df["Return_1D"].shift(1)
    df["Return_1D_Lag2"] = df["Return_1D"].shift(2)
    df["Return_1D_Lag5"] = df["Return_1D"].shift(5)

    return df


def add_target(df, horizon=1):
    df = df.copy()

    df["target_return"] = df["Return_1D"].shift(-horizon)

    return df

def add_market_context_features(df, benchmark_df=None, vix_df=None):
    df = df.copy()

    if benchmark_df is not None:
        benchmark = benchmark_df[["Close"]].copy()
        benchmark["Benchmark_Return_1D"] = benchmark["Close"].pct_change()

        df = df.join(
            benchmark[["Benchmark_Return_1D"]],
            how="left",
        )

    if vix_df is not None:
        vix = vix_df[["Close"]].copy()
        vix["VIX_Level"] = vix["Close"] / 100.0

        df = df.join(
            vix[["VIX_Level"]],
            how="left",
        )

    return df

def build_features(df, benchmark_df=None, vix_df=None):
    df = df.copy()

    df = add_return_features(df)
    df = add_moving_average_features(df)
    df = add_macd_features(df)
    df = add_rsi_features(df)
    df = add_stochastic_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_market_context_features(
        df,
        benchmark_df=benchmark_df,
        vix_df=vix_df,
    )
    df = add_target(df)

    return df

def prepare_modeling_data(df, benchmark_df=None, vix_df=None):
    df = build_features(
        df,
        benchmark_df=benchmark_df,
        vix_df=vix_df,
    )

    df = df.dropna().copy()

    X = df.drop(columns=["target_return"])
    y = df["target_return"]

    return X, y, df