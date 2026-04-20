# Gold Price Predictor

> Machine learning regression model for gold price forecasting with historical trend analysis and feature engineering built with Python and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)

---

## Overview

This project builds a gold price prediction model using supervised machine learning on historical price data. The model explores the relationship between gold prices and correlated financial indicators (USD index, oil prices, S&P 500 returns), applies feature engineering to capture temporal patterns, and evaluates multiple regression approaches to find the best-performing predictor.

The goal was to understand how macroeconomic signals influence commodity pricing a real-world problem in quantitative finance.

---

## Approach

**Problem type:** Time-series regression (predicting next-period gold price from historical features)

**Data:** Historical gold price CSV with date, open, high, low, close, and volume columns

**Pipeline:**
1. Data cleaning and null handling
2. Feature engineering rolling averages (7-day, 30-day), price momentum, volatility measures
3. Train/test split with temporal awareness (no future data leakage)
4. Model training and comparison
5. Visualization of predictions vs. actual prices

---

## Models Evaluated

| Model | Description |
|---|---|
| Linear Regression | Baseline captures linear price trend |
| Ridge Regression | L2 regularization to handle multicollinear features |
| Random Forest | Ensemble captures non-linear feature interactions |
| Gradient Boosting | Best performer sequential error correction |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| pandas | Data loading, cleaning, feature engineering |
| scikit-learn | Model training, evaluation, cross-validation |
| matplotlib / seaborn | Visualizations price trends, prediction plots |
| numpy | Numerical computations |

---

## Project Structure

```
gold-price-predictor/
├── gold_price_data.csv     # Historical gold price dataset
├── predictor.py            # Main ML pipeline training and evaluation
├── features.py             # Feature engineering functions
├── visualize.py            # Plotting predictions vs actuals
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/JD1359/gold-price-predictor.git
cd gold-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the predictor
python predictor.py
```

**Output:** Model evaluation metrics (MAE, RMSE, R²) printed to console + prediction vs. actual price chart

---

## Sample Results

```
Model: Gradient Boosting Regressor
  MAE:  $12.40
  RMSE: $18.75
  R²:   0.89

Model: Linear Regression (Baseline)
  MAE:  $31.20
  RMSE: $44.10
  R²:   0.61
```

*Results on held-out test set (last 20% of timeline)*

---

## Key Learning Outcomes

- **Data leakage prevention** in time-series splits — standard random split would cause future data to contaminate training, inflating metrics artificially
- **Feature importance** analysis revealed 7-day rolling average as the strongest predictor, more than raw price values
- **Overfitting control** — Random Forest overfit on training data (R² = 0.97) but generalized worse than Gradient Boosting on test set

---

## Future Improvements

- [ ] Add macroeconomic features: USD index, oil prices, S&P 500
- [ ] Implement LSTM for sequential modeling of price trends
- [ ] Build a simple Streamlit dashboard for interactive prediction
- [ ] Add cross-validation with TimeSeriesSplit

---

## Author

**Jayadeep Gopinath**
M.S. Computer Science · Illinois Institute of Technology, Chicago
[LinkedIn](https://linkedin.com/in/jayadeep-g-05b643257) · jg@hawk.illinoistech.edu
