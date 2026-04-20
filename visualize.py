import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "actual":    "#2563eb",
    "predicted": "#f59e0b",
    "error":     "#ef4444",
    "positive":  "#10b981",
    "negative":  "#ef4444",
}


def plot_predictions(dates, actual, predicted, model_name: str = "Model", save_path: str = None):
    """
    Plot actual vs predicted gold prices over time.

    Args:
        dates:      DatetimeIndex or list of dates (test period).
        actual:     Array of actual prices.
        predicted:  Array of predicted prices.
        model_name: Label for the model in the plot title.
        save_path:  If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Gold Price Prediction — {model_name}", fontsize=14, fontweight="bold", y=1.01)

    # ── Top: Actual vs Predicted ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, actual,    color=COLORS["actual"],    linewidth=2,   label="Actual Price",    zorder=3)
    ax1.plot(dates, predicted, color=COLORS["predicted"], linewidth=1.5, label="Predicted Price",linestyle="--", alpha=0.85, zorder=2)

    ax1.fill_between(dates, actual, predicted,where=(actual >= predicted),alpha=0.08, color=COLORS["actual"], label="Overestimate")
    ax1.fill_between(dates, actual, predicted,where=(actual < predicted),alpha=0.08, color=COLORS["predicted"], label="Underestimate")

    ax1.set_ylabel("Gold Price (USD)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.set_title("Actual vs Predicted (Test Set)", fontsize=11, pad=8)

    # ── Bottom: Residuals ────────────────────────────────────────────────────
    ax2 = axes[1]
    residuals = np.array(actual) - np.array(predicted)
    colors = [COLORS["positive"] if r >= 0 else COLORS["negative"] for r in residuals]
    ax2.bar(dates, residuals, color=colors, width=1.5, alpha=0.7)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residual (USD)", fontsize=10)
    ax2.set_title("Prediction Errors", fontsize=10, pad=6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Chart saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_model_comparison(results: list, save_path: str = None):
    """
    Bar chart comparing MAE, RMSE, and R² across all models.

    Args:
        results:   List of result dicts from predictor.evaluate_model().
        save_path: If provided, saves the figure to this path.
    """
    names = [r["name"] for r in results]
    mae   = [r["MAE"]  for r in results]
    rmse  = [r["RMSE"] for r in results]
    r2    = [r["R2"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Gold Price Prediction", fontsize=13, fontweight="bold")

    bar_colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    for ax, values, title, fmt in zip(
        axes,
        [mae, rmse, r2],
        ["MAE (lower is better)", "RMSE (lower is better)", "R² Score (higher is better)"],
        ["${:.2f}", "${:.2f}", "{:.4f}"]
    ):
        bars = ax.bar(names, values, color=bar_colors, alpha=0.85, width=0.5, edgecolor="white")
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Comparison chart saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(model, feature_cols: list, top_n: int = 12, save_path: str = None):
    """
    Horizontal bar chart of feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("  Model does not have feature_importances_. Skipping plot.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importances.index[::-1], importances.values[::-1],color="#3b82f6", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Feature Importance Score", fontsize=10)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, importances.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Feature importance chart saved: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Quick demo with dummy data
    import pandas as pd
    dates     = pd.date_range("2023-01-01", periods=100, freq="D")
    actual    = 1800 + np.cumsum(np.random.randn(100) * 5)
    predicted = actual + np.random.randn(100) * 12

    dummy_results = [
        {"name": "Linear Regression", "MAE": 31.2, "RMSE": 44.1, "R2": 0.61},
        {"name": "Ridge Regression",  "MAE": 28.5, "RMSE": 40.3, "R2": 0.67},
        {"name": "Random Forest",     "MAE": 15.8, "RMSE": 22.4, "R2": 0.85},
        {"name": "Gradient Boosting", "MAE": 12.4, "RMSE": 18.7, "R2": 0.89},
    ]

    plot_predictions(dates, actual, predicted, model_name="Gradient Boosting")
    plot_model_comparison(dummy_results)
