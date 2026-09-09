import re

import pandas as pd


def clean_text(text):
    """
    Clean news text before sentiment analysis.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs.
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def combine_headline_summary(headline, summary=""):
    """
    Combine headline and summary into one cleaned text field.
    """
    headline = clean_text(headline)
    summary = clean_text(summary)

    if headline and summary:
        return f"{headline}. {summary}"

    return headline or summary


def prepare_news_dataframe(
    news_df,
    timestamp_column="timestamp",
    headline_column="headline",
    summary_column="summary",
):
    """
    Clean and prepare a news DataFrame.
    """
    df = news_df.copy()

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if headline_column not in df.columns:
        raise ValueError(
            f"Missing headline column: {headline_column}"
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    df = df.dropna(subset=[timestamp_column]).copy()

    if summary_column not in df.columns:
        df[summary_column] = ""

    df["clean_text"] = df.apply(
        lambda row: combine_headline_summary(
            row[headline_column],
            row[summary_column],
        ),
        axis=1,
    )

    df = df[df["clean_text"].str.len() > 0].copy()

    return df


def vader_sentiment(text):
    """
    Calculate VADER compound sentiment score.

    Returns a score between -1 and +1.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "VADER is not installed. Install nltk and download "
            "the VADER lexicon."
        ) from exc

    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(str(text))

    return scores["compound"]


def add_vader_sentiment(
    news_df,
    text_column="clean_text",
):
    """
    Add VADER sentiment scores to a news DataFrame.
    """
    df = news_df.copy()

    if text_column not in df.columns:
        raise ValueError(
            f"Missing text column: {text_column}"
        )

    analyzer = None

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()

    except ImportError as exc:
        raise ImportError(
            "VADER is not installed. Install nltk."
        ) from exc

    except LookupError as exc:
        raise LookupError(
            "VADER lexicon is missing. Run "
            "nltk.download('vader_lexicon')."
        ) from exc

    df["VADER_Compound"] = df[text_column].apply(
        lambda text: analyzer.polarity_scores(
            str(text)
        )["compound"]
    )

    df["VADER_Label"] = df["VADER_Compound"].apply(
        lambda score: (
            "Positive"
            if score >= 0.05
            else "Negative"
            if score <= -0.05
            else "Neutral"
        )
    )

    return df


def aggregate_daily_sentiment(
    news_df,
    timestamp_column="timestamp",
):
    """
    Aggregate news sentiment by calendar day.
    """
    df = news_df.copy()

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if "VADER_Compound" not in df.columns:
        raise ValueError(
            "VADER_Compound column is required."
        )

    df["Date"] = pd.to_datetime(
        df[timestamp_column]
    ).dt.date

    daily = (
        df.groupby("Date")
        .agg(
            Sentiment_Mean=(
                "VADER_Compound",
                "mean",
            ),
            Sentiment_Std=(
                "VADER_Compound",
                "std",
            ),
            News_Volume=(
                "VADER_Compound",
                "count",
            ),
        )
        .reset_index()
    )

    daily["Sentiment_Std"] = (
        daily["Sentiment_Std"].fillna(0)
    )

    daily["Date"] = pd.to_datetime(
        daily["Date"]
    )

    return daily.set_index("Date").sort_index()