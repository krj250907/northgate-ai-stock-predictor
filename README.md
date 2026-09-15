# Northgate AI Stock Predictor

## Quantitative Research · AI Stock Prediction & Portfolio Optimization System

An end-to-end quantitative research project combining machine learning, deep learning, financial mathematics, sentiment analysis, portfolio optimization, backtesting, and an interactive Streamlit dashboard.

The system is designed as a reproducible research pipeline rather than a simple stock-price prediction model.

---

## 1. Project Overview

Financial markets are noisy, non-stationary, and difficult to predict consistently.

This project investigates whether historical market information, technical indicators, market context, deep-learning sequence models, and financial-news sentiment can provide useful signals for:

- Next-day stock-return prediction
- Directional prediction
- Risk analysis
- Portfolio optimization
- Portfolio backtesting
- Buy/Hold/Sell recommendations
- Portfolio rebalancing

The project evaluates both successful and unsuccessful modelling approaches and reports their limitations honestly.

---

## 2. Objectives

The main objectives are:

1. Build a reproducible financial-data ingestion pipeline.
2. Clean and align multiple market datasets.
3. Engineer technical and market-context features.
4. Perform exploratory and statistical analysis.
5. Establish a naive random-walk baseline.
6. Compare classical machine-learning models.
7. Compare eight deep-learning model families.
8. Analyze financial-news sentiment using VADER and FinBERT.
9. Measure the effect of sentiment on forecasting.
10. Implement financial mathematics from scratch.
11. Construct optimized portfolios using Modern Portfolio Theory.
12. Perform Monte Carlo portfolio simulation.
13. Backtest portfolio strategies out-of-sample.
14. Build a rule-based investment recommendation engine.
15. Evaluate recommendation performance using historical data.
16. Provide an interactive Streamlit dashboard.
17. Provide a reproducible end-to-end pipeline.

---

## 3. System Architecture

```text
                ┌──────────────────────┐
                │   Yahoo Finance      │
                │   Market Data        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      Ingestion       │
                │     src/ingest.py    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Cleaning & Alignment │
                │     src/clean.py     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Feature Engineering  │
                │   src/features.py    │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌─────────────────┐      ┌──────────────────┐
      │ Classical ML    │      │ Deep Learning    │
      │ Linear/Ridge    │      │ LSTM / GRU       │
      │ RF/XGBoost/SVR  │      │ BiLSTM/Transformer│
      └────────┬────────┘      │ CNN / Attention  │
               │               │ MLP              │
               │               └────────┬─────────┘
               │                        │
               └────────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Forecast Evaluation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Sentiment Analysis  │
                 │ VADER + FinBERT     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Portfolio Analytics │
                 │ MPT + Monte Carlo   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Recommendation      │
                 │ BUY / HOLD / SELL   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 └─────────────────────┘