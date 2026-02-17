import io
import base64
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import Response

import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="Heavy FastAPI App", version="1.0.0")


@app.get("/")
def hello():
    return {"message": "Hello, World!"}


@app.get("/numpy/random-matrix")
def numpy_random_matrix(
    rows: int = Query(5, ge=1, le=100), cols: int = Query(5, ge=1, le=100)
):
    """Generate a random matrix and return its stats."""
    matrix = np.random.rand(rows, cols)
    return {
        "shape": matrix.shape,
        "mean": float(matrix.mean()),
        "std": float(matrix.std()),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "matrix": matrix.tolist(),
    }


@app.get("/pandas/summary")
def pandas_summary(n: int = Query(50, ge=1, le=1000)):
    """Generate a random DataFrame and return summary statistics."""
    df = pd.DataFrame(
        {
            "price": np.random.uniform(10, 500, n),
            "quantity": np.random.randint(1, 100, n),
            "rating": np.random.uniform(1, 5, n),
            "category": np.random.choice(["A", "B", "C", "D"], n),
        }
    )
    df["revenue"] = df["price"] * df["quantity"]
    summary = df.describe().to_dict()
    group_stats = df.groupby("category")["revenue"].mean().to_dict()
    return {
        "rows": n,
        "summary": summary,
        "avg_revenue_by_category": group_stats,
    }


@app.get("/scipy/distribution")
def scipy_distribution(
    dist: str = Query("norm", enum=["norm", "uniform", "expon"]),
    n: int = Query(1000, ge=10, le=10000),
):
    """Generate samples from a distribution and run normality test."""
    if dist == "norm":
        samples = stats.norm.rvs(loc=0, scale=1, size=n)
    elif dist == "uniform":
        samples = stats.uniform.rvs(loc=0, scale=10, size=n)
    else:
        samples = stats.expon.rvs(scale=2, size=n)

    shapiro_stat, shapiro_p = stats.shapiro(samples[: min(n, 5000)])
    ks_stat, ks_p = stats.kstest(samples, "norm")

    return {
        "distribution": dist,
        "n_samples": n,
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "skew": float(stats.skew(samples)),
        "kurtosis": float(stats.kurtosis(samples)),
        "shapiro_test": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
        "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
    }


@app.get("/pillow/image")
def pillow_image(
    width: int = Query(256, ge=16, le=1024),
    height: int = Query(256, ge=16, le=1024),
    color: str = Query("blue"),
):
    """Generate a simple geometric image and return it as PNG."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, width - 10, height - 10], outline=color, width=3)
    draw.ellipse([30, 30, width - 30, height - 30], outline="red", width=2)
    draw.line([(0, 0), (width, height)], fill="green", width=2)
    draw.line([(width, 0), (0, height)], fill="green", width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/sklearn/regression")
def sklearn_regression(
    n: int = Query(100, ge=10, le=10000), noise: float = Query(1.0, ge=0)
):
    """Fit a linear regression on random data."""
    X = np.random.rand(n, 1) * 10
    y = 3.5 * X.flatten() + 7.2 + np.random.randn(n) * noise

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = model.score(X, y)
    residuals = y - y_pred

    return {
        "n_samples": n,
        "noise": noise,
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(r2),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "true_coefficient": 3.5,
        "true_intercept": 7.2,
    }


@app.get("/matplotlib/chart")
def matplotlib_chart(
    chart_type: str = Query("line", enum=["line", "bar", "scatter"]),
    n: int = Query(20, ge=5, le=100),
):
    """Generate a chart and return it as a PNG image."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n)
    y = np.cumsum(np.random.randn(n))

    if chart_type == "line":
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_title("Random Walk")
    elif chart_type == "bar":
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in y]
        ax.bar(x, y, color=colors)
        ax.set_title("Random Values")
    else:
        ax.scatter(x, y, c=y, cmap="viridis", s=80, edgecolors="black")
        ax.set_title("Random Scatter")

    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
def health():
    """Health check with dependency versions."""
    return {
        "status": "ok",
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.scipy.__version__
            if hasattr(stats, "scipy")
            else "installed",
            "Pillow": Image.__version__,
            "sklearn": "installed",
            "matplotlib": matplotlib.__version__,
        },
    }
