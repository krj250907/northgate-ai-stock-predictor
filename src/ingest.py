import os

import pandas as pd
import yfinance as yf
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def download_data(ticker, start_date):
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        start=start_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Flatten yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"

    return data


def main():
    config = load_config()

    start_date = config["data"]["start_date"]
    tickers = config["universe"]["stocks"]
    benchmark = config["universe"]["benchmark"]
    vix = config["universe"]["vix"]

    os.makedirs("data/raw", exist_ok=True)

    all_tickers = tickers + [benchmark, vix]

    for ticker in all_tickers:
        data = download_data(ticker, start_date)

        filename = ticker.replace("^", "") + ".csv"
        output_path = os.path.join("data/raw", filename)

        data.to_csv(output_path)

        print(f"Saved: {output_path} | Rows: {len(data)}")


if __name__ == "__main__":
    main()