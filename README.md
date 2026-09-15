# Northgate AI Stock Predictor

## Quantitative Research · AI Stock Prediction & Portfolio Optimization System

Northgate AI Stock Predictor is an end-to-end quantitative research project that combines financial time-series analysis, classical machine learning, deep learning, financial sentiment analysis, portfolio optimization, risk analytics, and rule-based investment recommendations.

The project is designed as an individual advanced capstone and emphasizes chronological validation, reproducibility, and prevention of look-ahead bias.

---

## 1. Project Objectives

The system aims to:

- Collect historical financial market data.
- Clean and align multi-asset time-series data.
- Engineer technical, market, volatility, and calendar features.
- Analyze financial time-series properties through EDA and statistical tests.
- Compare classical machine-learning forecasting models.
- Compare deep-learning sequence models.
- Incorporate financial news sentiment.
- Calculate portfolio risk and performance metrics.
- Optimize portfolios using Modern Portfolio Theory.
- Backtest portfolio strategies using out-of-sample data.
- Generate BUY/HOLD/SELL recommendations.
- Evaluate recommendation performance honestly against a baseline.
- Provide an interactive Streamlit dashboard.

---

## 2. System Architecture

```text
Yahoo Finance / FRED / Financial News
                │
                ▼
        Data Ingestion
                │
                ▼
       Data Cleaning & Alignment
                │
                ▼
        Feature Engineering
                │
        ┌───────┴────────┐
        ▼                ▼
   Classical ML       Deep Learning
        │                │
        └───────┬────────┘
                ▼
       Forecast Evaluation
                │
                ▼
        Sentiment Analysis
        FinBERT + VADER
                │
                ▼
       Risk & Portfolio Layer
                │
                ▼
       Recommendation Engine
                │
                ▼
        Streamlit Dashboard
