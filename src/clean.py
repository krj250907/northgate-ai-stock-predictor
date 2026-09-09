import os

import pandas as pd


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

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
VIX = "VIX"


def load_csv(ticker):
    path = os.path.join(RAW_DIR, f"{ticker}.csv")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    return df


def clean_data(df):
    df = df[~df.index.duplicated(keep="first")]

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Close"])

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_data = {}

    for ticker in TICKERS + [BENCHMARK, VIX]:
        print(f"Loading {ticker}...")

        df = load_csv(ticker)
        df = clean_data(df)

        all_data[ticker] = df

    # Find dates common to every asset
    common_dates = None

    for df in all_data.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    common_dates = common_dates.sort_values()

    print(f"\nCommon trading dates: {len(common_dates)}")
    print(f"First date: {common_dates.min().date()}")
    print(f"Last date: {common_dates.max().date()}")

    # Save aligned individual files
    for ticker, df in all_data.items():
        aligned = df.loc[common_dates]

        output_path = os.path.join(
            PROCESSED_DIR,
            f"{ticker}.csv",
        )

        aligned.to_csv(output_path)

        print(f"Saved: {output_path} | Rows: {len(aligned)}")


if __name__ == "__main__":
    main()