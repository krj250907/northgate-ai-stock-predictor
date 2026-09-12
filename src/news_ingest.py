import json
from pathlib import Path

import pandas as pd


NEWS_DIR = Path("data/raw/news")

TICKERS = {
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "NVDA",
    "KO",
    "HD",
}


def load_year(year):
    """Load one yearly FNSPID JSON file."""
    path = NEWS_DIR / f"{year}_processed.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_articles(years=None):
    """Load and filter articles mentioning our project tickers."""
    if years is None:
        years = range(2019, 2024)

    rows = []

    for year in years:
        articles = load_year(year)

        for article in articles:
            mentioned = set(article.get("mentioned_companies") or [])
            matched_tickers = sorted(mentioned.intersection(TICKERS))

            if not matched_tickers:
                continue

            rows.append(
                {
                    "date_publish": article.get("date_publish"),
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "maintext": article.get("maintext"),
                    "source_domain": article.get("source_domain"),
                    "url": article.get("url"),
                    "mentioned_companies": matched_tickers,
                    "news_outlet": article.get("news_outlet"),
                }
            )

    return pd.DataFrame(rows)


def explode_tickers(df):
    """Create one row per article-ticker pair."""
    df = df.copy()
    df = df.explode("mentioned_companies")
    df = df.rename(columns={"mentioned_companies": "ticker"})
    return df


def clean_news_dataframe(df):
    """Clean and standardize the news dataframe."""
    df = df.copy()

    df["date_publish"] = pd.to_datetime(
        df["date_publish"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["date_publish", "ticker", "title"])

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["maintext"] = df["maintext"].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(
        subset=["date_publish", "ticker", "title"]
    )

    df = df.sort_values(["ticker", "date_publish"]).reset_index(drop=True)

    return df


def build_news_dataset():
    """Build the cleaned historical news dataset."""
    df = extract_articles()
    df = explode_tickers(df)
    df = clean_news_dataframe(df)

    output_path = NEWS_DIR / "news_filtered.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Unique articles: {df['url'].nunique()}")
    print(f"Tickers: {sorted(df['ticker'].unique())}")
    print(
        f"Date range: "
        f"{df['date_publish'].min()} to {df['date_publish'].max()}"
    )

    return df


if __name__ == "__main__":
    build_news_dataset()