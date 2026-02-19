"""
Data analytics API -- lightweight, numpy-only (no pandas/scipy/sklearn/matplotlib).

Provides endpoints for sales analysis, forecasting, A/B testing,
customer segmentation, anomaly detection, and correlation matrices.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from fastapi import FastAPI, Query

app = FastAPI(title="Analytics API", version="1.0.0")


# ---------------------------------------------------------------------------
# Sales data generator (shared helper)
# ---------------------------------------------------------------------------


def _generate_sales_data(days: int, seed: int = 42) -> dict:
    """Generate realistic-looking daily sales data as plain numpy arrays + lists."""
    rng = np.random.default_rng(seed)
    end = datetime.utcnow().date()
    dates = [end - timedelta(days=days - 1 - i) for i in range(days)]

    trend = np.linspace(100, 150, days)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = rng.normal(0, 10, days)
    revenue = np.clip(trend + seasonality + noise, 10, None).round(2)

    categories = rng.choice(["Electronics", "Clothing", "Food", "Home"], days).tolist()
    regions = rng.choice(["North", "South", "East", "West"], days).tolist()
    units = rng.integers(5, 200, days)

    return {
        "dates": dates,
        "revenue": revenue,
        "units_sold": units,
        "categories": categories,
        "regions": regions,
    }


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using cumsum."""
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    out = (cs[window:] - cs[:-window]) / window
    return out


def _linreg(x: np.ndarray, y: np.ndarray):
    """Ordinary least-squares fit. Returns (slope, intercept)."""
    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sx2 = (x * x).sum()
    denom = n * sx2 - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return float(slope), float(intercept)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Groupby helper (replaces pandas groupby)
# ---------------------------------------------------------------------------


def _group_agg(keys: list, values: np.ndarray, agg: str = "sum") -> dict:
    """Group values by string keys and aggregate."""
    buckets: dict[str, list[float]] = {}
    for k, v in zip(keys, values):
        buckets.setdefault(k, []).append(float(v))
    result = {}
    for k, vals in sorted(buckets.items()):
        arr = np.array(vals)
        if agg == "sum":
            result[k] = round(float(arr.sum()), 2)
        elif agg == "mean":
            result[k] = round(float(arr.mean()), 2)
    return result


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
            "/analytics/anomaly-detection",
            "/analytics/correlation-matrix",
        ],
    }


@app.get("/analytics/sales-summary")
def sales_summary(days: int = Query(90, ge=7, le=730)):
    """Aggregate sales data and return KPIs."""
    data = _generate_sales_data(days)
    rev = data["revenue"]

    total_revenue = round(float(rev.sum()), 2)
    avg_daily = round(float(rev.mean()), 2)
    std_daily = round(float(rev.std()), 2)

    best_idx = int(rev.argmax())
    worst_idx = int(rev.argmin())

    by_category = {}
    for cat in sorted(set(data["categories"])):
        mask = [c == cat for c in data["categories"]]
        cat_rev = rev[mask]
        cat_units = data["units_sold"][mask]
        by_category[cat] = {
            "total_revenue": round(float(cat_rev.sum()), 2),
            "avg_units": round(float(cat_units.mean()), 2),
        }

    by_region = _group_agg(data["regions"], rev, "sum")

    rolling = _rolling_mean(rev, 7)
    rolling_last5 = [round(float(v), 2) for v in rolling[-5:]]

    return {
        "period_days": days,
        "total_revenue": total_revenue,
        "avg_daily_revenue": avg_daily,
        "std_daily_revenue": std_daily,
        "best_day": {
            "date": str(data["dates"][best_idx]),
            "revenue": float(rev[best_idx]),
        },
        "worst_day": {
            "date": str(data["dates"][worst_idx]),
            "revenue": float(rev[worst_idx]),
        },
        "by_category": by_category,
        "by_region": by_region,
        "rolling_7d_last5": rolling_last5,
    }


@app.get("/analytics/forecast")
def forecast(
    days_history: int = Query(90, ge=30, le=365),
    days_ahead: int = Query(14, ge=1, le=60),
):
    """Fit a linear regression on historical sales and forecast future revenue."""
    data = _generate_sales_data(days_history)
    rev = data["revenue"]

    x = np.arange(len(rev), dtype=np.float64)
    slope, intercept = _linreg(x, rev)

    y_pred = slope * x + intercept
    r2_val = _r2(rev, y_pred)
    rmse_val = _rmse(rev, y_pred)

    last_date = data["dates"][-1]
    forecast_list = []
    for i in range(1, days_ahead + 1):
        d = last_date + timedelta(days=i)
        v = slope * (len(rev) - 1 + i) + intercept
        forecast_list.append({"date": str(d), "predicted_revenue": round(v, 2)})

    return {
        "model": "LinearRegression (numpy)",
        "history_days": days_history,
        "r_squared": round(r2_val, 4),
        "rmse": round(rmse_val, 2),
        "coefficient": round(slope, 4),
        "intercept": round(intercept, 2),
        "forecast": forecast_list,
    }


@app.get("/analytics/ab-test")
def ab_test(
    n_a: int = Query(500, ge=30, le=50000),
    n_b: int = Query(500, ge=30, le=50000),
    conv_rate_a: float = Query(0.10, ge=0.001, le=0.999),
    conv_rate_b: float = Query(0.12, ge=0.001, le=0.999),
):
    """Simulate an A/B test and run a two-proportion z-test."""
    rng = np.random.default_rng(42)
    a = rng.binomial(1, conv_rate_a, n_a)
    b = rng.binomial(1, conv_rate_b, n_b)

    obs_rate_a = float(a.mean())
    obs_rate_b = float(b.mean())

    p_pool = (a.sum() + b.sum()) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if p_pool > 0 else 1e-9
    z_stat = (obs_rate_b - obs_rate_a) / se
    p_value = 2 * (1 - _norm_cdf(abs(z_stat)))

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
            "z_statistic": round(z_stat, 4),
            "p_value": round(p_value, 6),
        },
        "lift_pct": round(lift, 2) if lift is not None else None,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


@app.get("/analytics/anomaly-detection")
def anomaly_detection(
    days: int = Query(90, ge=14, le=365),
    threshold: float = Query(2.0, ge=1.0, le=4.0),
):
    """Detect anomalous days in sales data using z-score method."""
    data = _generate_sales_data(days)
    rev = data["revenue"]

    mean_rev = float(rev.mean())
    std_rev = float(rev.std())
    z_scores = (rev - mean_rev) / std_rev if std_rev > 0 else np.zeros_like(rev)

    anomalies = []
    for i in range(len(rev)):
        if abs(z_scores[i]) > threshold:
            anomalies.append(
                {
                    "date": str(data["dates"][i]),
                    "revenue": round(float(rev[i]), 2),
                    "z_score": round(float(z_scores[i]), 3),
                }
            )

    return {
        "period_days": days,
        "threshold_sigma": threshold,
        "mean_revenue": round(mean_rev, 2),
        "std_revenue": round(std_rev, 2),
        "total_anomalies": len(anomalies),
        "anomaly_rate_pct": round(len(anomalies) / days * 100, 2),
        "anomalies": anomalies,
    }


@app.get("/analytics/correlation-matrix")
def correlation_matrix(days: int = Query(90, ge=14, le=365)):
    """Compute the correlation matrix of sales metrics."""
    data = _generate_sales_data(days)

    day_of_week = np.array([d.weekday() for d in data["dates"]], dtype=np.float64)
    is_weekend = (day_of_week >= 5).astype(np.float64)

    cols = {
        "revenue": data["revenue"].astype(np.float64),
        "units_sold": data["units_sold"].astype(np.float64),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }
    names = list(cols.keys())
    matrix = np.column_stack([cols[n] for n in names])
    corr = np.corrcoef(matrix, rowvar=False)

    corr_dict = {}
    for i, ni in enumerate(names):
        corr_dict[ni] = {nj: round(float(corr[i, j]), 4) for j, nj in enumerate(names)}

    return {
        "period_days": days,
        "columns": names,
        "correlation_matrix": corr_dict,
    }
