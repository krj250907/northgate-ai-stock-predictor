import re

import pandas as pd


# ============================================================
# TEXT CLEANING
# ============================================================

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

    df = df.dropna(
        subset=[timestamp_column]
    ).copy()

    if summary_column not in df.columns:
        df[summary_column] = ""

    df["clean_text"] = df.apply(
        lambda row: combine_headline_summary(
            row[headline_column],
            row[summary_column],
        ),
        axis=1,
    )

    df = df[
        df["clean_text"].str.len() > 0
    ].copy()

    return df


# ============================================================
# VADER SENTIMENT
# ============================================================

def vader_sentiment(text):
    """
    Calculate VADER compound sentiment score.

    Returns a score between -1 and +1.
    """

    try:
        from nltk.sentiment.vader import (
            SentimentIntensityAnalyzer,
        )
    except ImportError as exc:
        raise ImportError(
            "VADER is not installed. Install nltk and "
            "download the VADER lexicon."
        ) from exc

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise LookupError(
            "VADER lexicon is missing. Run "
            "nltk.download('vader_lexicon')."
        ) from exc

    scores = analyzer.polarity_scores(
        str(text)
    )

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

    try:
        from nltk.sentiment.vader import (
            SentimentIntensityAnalyzer,
        )

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

    df["VADER_Compound"] = df[
        text_column
    ].apply(
        lambda text: analyzer.polarity_scores(
            str(text)
        )["compound"]
    )

    df["VADER_Label"] = df[
        "VADER_Compound"
    ].apply(
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
    Aggregate VADER news sentiment by calendar day.
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

    return (
        daily
        .set_index("Date")
        .sort_index()
    )


# ============================================================
# FINBERT SENTIMENT
# ============================================================

def finbert_sentiment(
    text,
    model_name="ProsusAI/finbert",
):
    """
    Calculate FinBERT financial sentiment.

    Returns:
        label: positive, negative, or neutral
        score: model confidence for the predicted label
    """

    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(model_name)
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_index = int(
        torch.argmax(probabilities)
    )

    labels = [
        model.config.id2label[i].lower()
        for i in range(
            len(model.config.id2label)
        )
    ]

    label = labels[predicted_index]

    score = float(
        probabilities[predicted_index]
    )

    return label, score


def add_finbert_sentiment(
    news_df,
    text_column="clean_text",
    model_name="ProsusAI/finbert",
):
    """
    Add FinBERT sentiment labels and confidence
    scores to a news DataFrame.
    """

    df = news_df.copy()

    if text_column not in df.columns:
        raise ValueError(
            f"Missing text column: {text_column}"
        )

    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(model_name)
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    labels = [
        model.config.id2label[i].lower()
        for i in range(
            len(model.config.id2label)
        )
    ]

    sentiment_labels = []
    sentiment_scores = []

    for text in df[text_column]:

        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predicted_index = int(
            torch.argmax(probabilities)
        )

        sentiment_labels.append(
            labels[predicted_index]
        )

        sentiment_scores.append(
            float(
                probabilities[predicted_index]
            )
        )

    df["FinBERT_Label"] = sentiment_labels
    df["FinBERT_Score"] = sentiment_scores

    return df


# ============================================================
# NEWS → TRADING SESSION
# ============================================================

def map_news_to_trading_session(
    news_df,
    trading_dates,
    timestamp_column="Datetime",
):
    """
    Map news timestamps to the trading session they can
    legitimately influence.

    News published after 4:00 PM ET is assigned to the
    next available trading session.

    News published before or during market hours is assigned
    to the current trading session.
    """

    df = news_df.copy()

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if len(df) == 0:
        df["Trading_Date"] = pd.Series(
            dtype="datetime64[ns]"
        )
        return df

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=[timestamp_column]
    ).copy()

    # Convert UTC timestamps to US Eastern time.
    df["Local_Datetime"] = (
        df[timestamp_column]
        .dt.tz_convert("America/New_York")
    )

    df["Trading_Date"] = (
        df["Local_Datetime"]
        .dt.normalize()
        .dt.tz_localize(None)
    )

    # After 4:00 PM ET → next trading session.
    after_close = (
        df["Local_Datetime"].dt.hour >= 16
    )

    df.loc[
        after_close,
        "Trading_Date",
    ] = (
        df.loc[
            after_close,
            "Trading_Date",
        ]
        + pd.Timedelta(days=1)
    )

    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(trading_dates)
    ).normalize()

    # Map each news date to the first available
    # trading date on or after that date.
    def next_trading_date(date):
        valid_dates = trading_dates[
            trading_dates >= date
        ]

        if len(valid_dates) > 0:
            return valid_dates[0]

        return pd.NaT

    df["Trading_Date"] = (
        pd.to_datetime(
            df["Trading_Date"]
        ).map(next_trading_date)
    )

    return df.dropna(
        subset=["Trading_Date"]
    ).copy()


# ============================================================
# MOMENTUM FEATURE
# ============================================================

def add_momentum_feature(
    sentiment_df,
    price_df,
    price_column="Close",
):
    """
    Add 3-day price momentum to daily sentiment data.

    IMPORTANT:
    Momentum uses only prices available BEFORE the
    corresponding trading session.

    The shift(1) prevents look-ahead leakage.
    """

    sentiment = sentiment_df.copy()
    prices = price_df.copy()

    prices.index = pd.to_datetime(
        prices.index
    )

    prices = prices.sort_index()

    if price_column not in prices.columns:
        raise ValueError(
            f"Missing price column: {price_column}"
        )

    # Calculate 3-day momentum and shift by one day
    # so today's feature only uses information known
    # before today's trading session.
    prices["Momentum_3D"] = (
        prices[price_column]
        .pct_change(3)
        .shift(1)
    )

    momentum = prices[
        "Momentum_3D"
    ].rename("Momentum_3D")

    sentiment["Trading_Date"] = (
        pd.to_datetime(
            sentiment["Trading_Date"]
        )
    )

    sentiment = sentiment.merge(
        momentum,
        left_on="Trading_Date",
        right_index=True,
        how="left",
    )

    return sentiment


# ============================================================
# NEWS VOLUME SPIKE
# ============================================================

def add_news_volume_spike(
    sentiment_df,
    window=20,
    threshold=2.0,
):
    """
    Detect unusually high daily news volume.

    A spike occurs when the current day's news volume
    is at least `threshold` times the rolling average
    of the previous `window` days.

    The rolling average excludes the current day
    to avoid look-ahead.

    If ticker is present, calculations are performed
    separately for each ticker.
    """

    df = sentiment_df.copy()

    if "News_Volume" not in df.columns:
        raise ValueError(
            "News_Volume column is required."
        )

    if "Trading_Date" not in df.columns:
        raise ValueError(
            "Trading_Date column is required."
        )

    df["Trading_Date"] = pd.to_datetime(
        df["Trading_Date"]
    )

    # --------------------------------------------------------
    # Ticker-specific calculation
    # --------------------------------------------------------

    if "ticker" in df.columns:

        df = df.sort_values(
            ["ticker", "Trading_Date"]
        ).copy()

        rolling_volume = (
            df.groupby("ticker")["News_Volume"]
            .transform(
                lambda x: (
                    x.shift(1)
                    .rolling(
                        window=window,
                        min_periods=5,
                    )
                    .mean()
                )
            )
        )

    else:

        df = df.sort_values(
            "Trading_Date"
        ).copy()

        rolling_volume = (
            df["News_Volume"]
            .shift(1)
            .rolling(
                window=window,
                min_periods=5,
            )
            .mean()
        )

    df["News_Volume_Rolling_Mean"] = (
        rolling_volume
    )

    df["News_Volume_Ratio"] = (
        df["News_Volume"]
        / df["News_Volume_Rolling_Mean"]
    )

    df["News_Volume_Spike"] = (
        df["News_Volume_Ratio"]
        >= threshold
    )

    return df


# ============================================================
# FINBERT DAILY AGGREGATION
# ============================================================

def aggregate_finbert_sentiment(df):
    """
    Aggregate FinBERT sentiment by ticker and trading date.

    Produces one row per ticker per trading day.

    FinBERT labels are converted into signed sentiment:

        positive -> +confidence
        neutral  -> 0
        negative -> -confidence
    """

    required_columns = [
        "ticker",
        "Trading_Date",
        "FinBERT_Score",
        "FinBERT_Label",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df.copy()

    data["Trading_Date"] = pd.to_datetime(
        data["Trading_Date"]
    )

    # Convert FinBERT prediction into signed sentiment.
    data["FinBERT_Sentiment"] = (
        data["FinBERT_Score"]
        * data["FinBERT_Label"].map(
            {
                "positive": 1.0,
                "neutral": 0.0,
                "negative": -1.0,
            }
        )
    )

    result = (
        data.groupby(
            ["ticker", "Trading_Date"]
        )
        .agg(
            FinBERT_Sentiment_Mean=(
                "FinBERT_Sentiment",
                "mean",
            ),
            FinBERT_Sentiment_Std=(
                "FinBERT_Sentiment",
                "std",
            ),
            FinBERT_News_Volume=(
                "FinBERT_Sentiment",
                "count",
            ),
        )
        .reset_index()
    )

    # A single article has zero dispersion.
    result["FinBERT_Sentiment_Std"] = (
        result["FinBERT_Sentiment_Std"]
        .fillna(0.0)
    )

    return (
        result
        .sort_values(
            ["ticker", "Trading_Date"]
        )
        .reset_index(drop=True)
    )