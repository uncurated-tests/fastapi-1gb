"""
Data analytics API — a second Vercel serverless function.

Provides endpoints for data analysis, statistical testing,
machine learning predictions, and report generation using
numpy, pandas, scipy, and scikit-learn.
"""

import io
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, Query, Body
from fastapi.responses import Response

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="Analytics API", version="1.0.0")


# ---------------------------------------------------------------------------
# Sales data generator (shared helper)
# ---------------------------------------------------------------------------


def _generate_sales_data(days: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-looking daily sales data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.utcnow().date(), periods=days, freq="D")
    trend = np.linspace(100, 150, days)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = rng.normal(0, 10, days)
    revenue = trend + seasonality + noise
    revenue = np.clip(revenue, 10, None)

    categories = rng.choice(["Electronics", "Clothing", "Food", "Home"], days)
    regions = rng.choice(["North", "South", "East", "West"], days)
    units = rng.integers(5, 200, days)

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": np.round(revenue, 2),
            "units_sold": units,
            "category": categories,
            "region": regions,
        }
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/analytics")
def analytics_root():
    return {
        "service": "Analytics API",
        "endpoints": [
            "/analytics/sales-summary",
            "/analytics/forecast",
            "/analytics/ab-test",
            "/analytics/segmentation",
            "/analytics/anomaly-detection",
            "/analytics/correlation-matrix",
            "/analytics/chart/sales-trend",
        ],
    }


@app.get("/analytics/sales-summary")
def sales_summary(days: int = Query(90, ge=7, le=730)):
    """Aggregate sales data and return KPIs."""
    df = _generate_sales_data(days)

    total_revenue = float(df["revenue"].sum())
    avg_daily_revenue = float(df["revenue"].mean())
    best_day = df.loc[df["revenue"].idxmax()]
    worst_day = df.loc[df["revenue"].idxmin()]

    by_category = (
        df.groupby("category")
        .agg(total_revenue=("revenue", "sum"), avg_units=("units_sold", "mean"))
        .round(2)
        .to_dict(orient="index")
    )

    by_region = df.groupby("region")["revenue"].sum().round(2).to_dict()

    rolling_7d = df.set_index("date")["revenue"].rolling(7).mean().dropna()

    return {
        "period_days": days,
        "total_revenue": round(total_revenue, 2),
        "avg_daily_revenue": round(avg_daily_revenue, 2),
        "std_daily_revenue": round(float(df["revenue"].std()), 2),
        "best_day": {
            "date": str(best_day["date"].date()),
            "revenue": float(best_day["revenue"]),
        },
        "worst_day": {
            "date": str(worst_day["date"].date()),
            "revenue": float(worst_day["revenue"]),
        },
        "by_category": by_category,
        "by_region": by_region,
        "rolling_7d_last5": rolling_7d.tail(5).round(2).tolist(),
    }


@app.get("/analytics/forecast")
def forecast(
    days_history: int = Query(90, ge=30, le=365),
    days_ahead: int = Query(14, ge=1, le=60),
):
    """Fit a linear regression on historical sales and forecast future revenue."""
    df = _generate_sales_data(days_history)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred_hist = model.predict(X)
    r2 = r2_score(y, y_pred_hist)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_hist)))

    X_future = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
    y_future = model.predict(X_future)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + timedelta(days=1), periods=days_ahead, freq="D"
    )

    return {
        "model": "LinearRegression",
        "history_days": days_history,
        "r_squared": round(float(r2), 4),
        "rmse": round(rmse, 2),
        "coefficient": round(float(model.coef_[0]), 4),
        "intercept": round(float(model.intercept_), 2),
        "forecast": [
            {"date": str(d.date()), "predicted_revenue": round(float(v), 2)}
            for d, v in zip(future_dates, y_future)
        ],
    }


@app.get("/analytics/ab-test")
def ab_test(
    n_a: int = Query(500, ge=30, le=50000),
    n_b: int = Query(500, ge=30, le=50000),
    conv_rate_a: float = Query(0.10, ge=0.001, le=0.999),
    conv_rate_b: float = Query(0.12, ge=0.001, le=0.999),
):
    """Simulate an A/B test and run statistical significance analysis."""
    rng = np.random.default_rng(42)
    a = rng.binomial(1, conv_rate_a, n_a)
    b = rng.binomial(1, conv_rate_b, n_b)

    obs_rate_a = float(a.mean())
    obs_rate_b = float(b.mean())

    # Two-proportion z-test
    p_pool = (a.sum() + b.sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (obs_rate_b - obs_rate_a) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Mann-Whitney U as a non-parametric alternative
    u_stat, u_pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")

    lift = ((obs_rate_b - obs_rate_a) / obs_rate_a * 100) if obs_rate_a > 0 else None

    return {
        "variant_a": {
            "n": n_a,
            "conversions": int(a.sum()),
            "rate": round(obs_rate_a, 4),
        },
        "variant_b": {
            "n": n_b,
            "conversions": int(b.sum()),
            "rate": round(obs_rate_b, 4),
        },
        "z_test": {
            "z_statistic": round(float(z_stat), 4),
            "p_value": round(p_value, 6),
        },
        "mann_whitney": {
            "u_statistic": float(u_stat),
            "p_value": round(float(u_pvalue), 6),
        },
        "lift_pct": round(lift, 2) if lift is not None else None,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


@app.get("/analytics/segmentation")
def customer_segmentation(
    n_customers: int = Query(500, ge=50, le=10000),
    n_segments: int = Query(4, ge=2, le=10),
):
    """Cluster customers using KMeans on synthetic RFM data."""
    rng = np.random.default_rng(42)

    recency = rng.exponential(30, n_customers)
    frequency = rng.poisson(5, n_customers) + 1
    monetary = rng.lognormal(4, 1, n_customers)

    X = np.column_stack([recency, frequency, monetary])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_segments, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = float(silhouette_score(X_scaled, labels))

    df = pd.DataFrame(
        {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "segment": labels,
        }
    )
    segment_profiles = (
        df.groupby("segment")
        .agg(
            count=("recency", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(2)
        .to_dict(orient="index")
    )

    return {
        "n_customers": n_customers,
        "n_segments": n_segments,
        "silhouette_score": round(sil_score, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "iterations": int(kmeans.n_iter_),
        "segment_profiles": segment_profiles,
    }


@app.get("/analytics/anomaly-detection")
def anomaly_detection(
    days: int = Query(90, ge=14, le=365),
    threshold: float = Query(2.0, ge=1.0, le=4.0),
):
    """Detect anomalous days in sales data using z-score method."""
    df = _generate_sales_data(days)

    mean_rev = df["revenue"].mean()
    std_rev = df["revenue"].std()
    df["z_score"] = (df["revenue"] - mean_rev) / std_rev
    df["is_anomaly"] = df["z_score"].abs() > threshold

    anomalies = df[df["is_anomaly"]][["date", "revenue", "z_score"]].copy()
    anomalies["date"] = anomalies["date"].dt.strftime("%Y-%m-%d")

    return {
        "period_days": days,
        "threshold_sigma": threshold,
        "mean_revenue": round(float(mean_rev), 2),
        "std_revenue": round(float(std_rev), 2),
        "total_anomalies": len(anomalies),
        "anomaly_rate_pct": round(len(anomalies) / len(df) * 100, 2),
        "anomalies": anomalies.round(3).to_dict(orient="records"),
    }


@app.get("/analytics/correlation-matrix")
def correlation_matrix(days: int = Query(90, ge=14, le=365)):
    """Compute the correlation matrix of sales metrics."""
    df = _generate_sales_data(days)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    numeric = df[["revenue", "units_sold", "day_of_week", "is_weekend"]]
    corr = numeric.corr().round(4)

    return {
        "period_days": days,
        "columns": list(corr.columns),
        "correlation_matrix": corr.to_dict(),
    }


@app.get("/analytics/chart/sales-trend")
def sales_trend_chart(
    days: int = Query(60, ge=7, le=365),
    show_forecast: bool = Query(True),
    forecast_days: int = Query(14, ge=1, le=60),
):
    """Generate a sales trend chart with optional forecast as PNG."""
    df = _generate_sales_data(days)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["revenue"],
        color="#2196F3",
        linewidth=1,
        alpha=0.6,
        label="Daily Revenue",
    )

    rolling = df.set_index("date")["revenue"].rolling(7).mean()
    ax.plot(
        rolling.index,
        rolling.values,
        color="#1565C0",
        linewidth=2,
        label="7-day Moving Avg",
    )

    if show_forecast:
        X = np.arange(len(df)).reshape(-1, 1)
        model = LinearRegression().fit(X, df["revenue"].values)

        X_future = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
        y_future = model.predict(X_future)
        future_dates = pd.date_range(
            start=df["date"].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )
        ax.plot(
            future_dates,
            y_future,
            color="#F44336",
            linewidth=2,
            linestyle="--",
            label="Forecast",
        )

    ax.set_title("Sales Revenue Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
