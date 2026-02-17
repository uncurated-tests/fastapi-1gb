import io
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import Response

import numpy as np
import pandas as pd
from scipy import stats, signal, fft
from PIL import Image, ImageDraw, ImageFilter
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


@app.get("/numpy/linalg")
def numpy_linalg(size: int = Query(4, ge=2, le=10)):
    """Generate a random square matrix and compute eigenvalues and determinant."""
    matrix = np.random.rand(size, size)
    eigenvalues = np.linalg.eigvals(matrix)
    det = np.linalg.det(matrix)
    inv = np.linalg.inv(matrix)
    return {
        "size": size,
        "determinant": float(det),
        "eigenvalues_real": [float(e.real) for e in eigenvalues],
        "eigenvalues_imag": [float(e.imag) for e in eigenvalues],
        "inverse_trace": float(np.trace(inv)),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
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


@app.get("/pandas/timeseries")
def pandas_timeseries(days: int = Query(30, ge=7, le=365)):
    """Generate a random timeseries and return rolling statistics."""
    dates = pd.date_range(start="2025-01-01", periods=days, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "value": np.cumsum(np.random.randn(days)) + 100,
            "volume": np.random.randint(1000, 50000, days),
        }
    )
    df["rolling_mean_7d"] = df["value"].rolling(7).mean()
    df["rolling_std_7d"] = df["value"].rolling(7).std()
    df["pct_change"] = df["value"].pct_change()
    return {
        "days": days,
        "start_value": float(df["value"].iloc[0]),
        "end_value": float(df["value"].iloc[-1]),
        "total_return_pct": float(
            (df["value"].iloc[-1] / df["value"].iloc[0] - 1) * 100
        ),
        "max_drawdown": float(df["value"].min() - df["value"].max()),
        "avg_daily_volume": float(df["volume"].mean()),
        "data": df.tail(10).to_dict(orient="records"),
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


@app.get("/pillow/blur")
def pillow_blur(
    width: int = Query(256, ge=16, le=512),
    height: int = Query(256, ge=16, le=512),
    radius: int = Query(5, ge=1, le=20),
):
    """Generate a noisy image and apply Gaussian blur."""
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

    buf = io.BytesIO()
    blurred.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


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


@app.get("/matplotlib/histogram")
def matplotlib_histogram(
    n: int = Query(1000, ge=100, le=10000),
    bins: int = Query(30, ge=5, le=100),
):
    """Generate a histogram of normally distributed data."""
    data = np.random.randn(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, edgecolor="black", alpha=0.7, color="#4CAF50")
    ax.set_title(f"Normal Distribution (n={n})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.axvline(
        np.mean(data), color="red", linestyle="--", label=f"Mean: {np.mean(data):.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


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


@app.get("/scipy/fft")
def scipy_fft_endpoint(
    n: int = Query(256, ge=16, le=4096),
    freq: float = Query(10.0, ge=1, le=100),
):
    """Generate a signal and compute its FFT."""
    t = np.linspace(0, 1, n, endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    sig += np.random.randn(n) * 0.3

    fft_vals = fft.fft(sig)
    freqs = fft.fftfreq(n, d=1.0 / n)
    magnitudes = np.abs(fft_vals)[: n // 2]
    freq_axis = freqs[: n // 2]

    peak_idx = np.argmax(magnitudes[1:]) + 1
    return {
        "n_samples": n,
        "input_freq": freq,
        "detected_peak_freq": float(freq_axis[peak_idx]),
        "peak_magnitude": float(magnitudes[peak_idx]),
        "top_5_freqs": [float(f) for f in freq_axis[np.argsort(magnitudes)[-5:][::-1]]],
        "top_5_magnitudes": [float(m) for m in np.sort(magnitudes)[-5:][::-1]],
    }


@app.get("/scipy/interpolate")
def scipy_interpolate(n: int = Query(10, ge=3, le=50)):
    """Generate random points and interpolate between them."""
    from scipy.interpolate import interp1d

    x = np.sort(np.random.rand(n)) * 10
    y = np.sin(x) + np.random.randn(n) * 0.2

    f_linear = interp1d(x, y, kind="linear")
    f_cubic = interp1d(x, y, kind="cubic")

    x_dense = np.linspace(x[0], x[-1], 100)
    y_linear = f_linear(x_dense)
    y_cubic = f_cubic(x_dense)

    return {
        "n_points": n,
        "original_x": x.tolist(),
        "original_y": y.tolist(),
        "interpolated_points": 100,
        "linear_range": {"min": float(y_linear.min()), "max": float(y_linear.max())},
        "cubic_range": {"min": float(y_cubic.min()), "max": float(y_cubic.max())},
        "max_linear_cubic_diff": float(np.max(np.abs(y_linear - y_cubic))),
    }


@app.get("/health")
def health():
    """Health check with dependency versions."""
    import scipy

    return {
        "status": "ok",
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "Pillow": Image.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
