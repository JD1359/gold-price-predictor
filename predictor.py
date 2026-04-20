import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from features import build_features, get_feature_columns


# ── Data Loading ────────────────────────────────────────────────────────────

def load_data(filepath: str = "gold_price_data.csv") -> pd.DataFrame:
    """
    Load gold price CSV. Expects columns: Date, Open, High, Low, Close, Volume.
    If the file has different column names, they are normalized here.
    """
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Parse date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "Date"

    df = df.sort_index()
    print(f" Loaded {len(df)} rows from {filepath}")
    print(f"   Date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


# ── Train/Test Split ─────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Temporal train/test split — last N% of data is test set.
    CRITICAL: Never shuffle time-series data — that leaks future data into training.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f" Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ── Model Training ────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return all models to evaluate."""
    return {
        "Linear Regression":      LinearRegression(),
        "Ridge Regression":       Ridge(alpha=1.0),
        "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,max_depth=4, random_state=42),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> dict:
    """Train a model and return evaluation metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    return {
        "name":       name,
        "model":      model,
        "predictions": preds,
        "MAE":        mae,
        "RMSE":       rmse,
        "R2":         r2,
        "MAPE":       mape,
    }


# ── Feature Importance ────────────────────────────────────────────────────────

def print_feature_importance(model, feature_cols: list, top_n: int = 10):
    """Print top N most important features for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(top_n)

    print(f"\nTop {top_n} Feature Importances:")
    for feat, imp in top.items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<20} {bar:<40} {imp:.4f}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(filepath: str = "gold_price_data.csv", test_ratio: float = 0.2):
    print("\n" + "═" * 60)
    print("  GOLD PRICE PREDICTOR — ML Pipeline")
    print("═" * 60)

    # 1. Load
    df_raw = load_data(filepath)

    # 2. Feature engineering
    print("\nEngineering features...")
    df = build_features(df_raw, price_col="Close", horizon=1)
    feature_cols = get_feature_columns(df, price_col="Close")
    print(f"   {len(feature_cols)} features generated")

    # 3. Split
    print("\nSplitting data (temporal — no shuffle)...")
    train, test = time_series_split(df, test_ratio)

    X_train = train[feature_cols]
    y_train = train["Target"]
    X_test  = test[feature_cols]
    y_test  = test["Target"]

    # 4. Scale features (important for linear models)
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 5. Train & evaluate all models
    print("\nTraining models...\n")
    models  = get_models()
    results = []

    for name, model in models.items():
        # Use scaled data for linear models, raw for tree models
        if "Regression" in name or "Ridge" in name:
            res = evaluate_model(name, model, X_train_sc, X_test_sc, y_train, y_test)
        else:
            res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # 6. Print results table
    print("─" * 60)
    print(f"  {'Model':<26} {'MAE':>8} {'RMSE':>9} {'R²':>7} {'MAPE':>8}")
    print("─" * 60)

    best = None
    for r in sorted(results, key=lambda x: x["R2"], reverse=True):
        marker = " ◄ BEST" if best is None else ""
        if best is None:
            best = r
        print(f"  {r['name']:<26} ${r['MAE']:>7.2f} ${r['RMSE']:>8.2f} {r['R2']:>7.4f} {r['MAPE']:>7.2f}%{marker}")

    print("─" * 60)

    # 7. Feature importance for best model
    print_feature_importance(best["model"], feature_cols)

    # 8. Sample predictions
    print(f"\nSample Predictions (last 5 test days):")
    sample_dates  = test.index[-5:]
    sample_actual = y_test.values[-5:]
    sample_preds  = best["predictions"][-5:]

    print(f"  {'Date':<14} {'Actual':>10} {'Predicted':>12} {'Error':>10}")
    print("  " + "─" * 48)
    for date, actual, pred in zip(sample_dates, sample_actual, sample_preds):
        error = actual - pred
        print(f"  {str(date.date()):<14} ${actual:>9.2f} ${pred:>11.2f} ${error:>+9.2f}")

    print(f"\nBest model: {best['name']} | R² = {best['R2']:.4f} | MAE = ${best['MAE']:.2f}")
    return best


if __name__ == "__main__":
    run("gold_price_data.csv")
