"""Correlate Kraken BTC OHLC data with Polymarket BTC market probabilities.

The script loads CSV exports from ``scripts/fetch_all_data.py`` (or compatible
snapshots), aligns the timestamps, computes several correlation metrics, and
produces a few exploratory plots.
"""
from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KRAKEN_PATH = ROOT_DIR / "data" / "raw" / "kraken_btc_ohlc.csv"
DEFAULT_POLYMARKET_PATH = ROOT_DIR / "data" / "raw" / "polymarket_btc_markets.csv"
PLOTS_DIR = ROOT_DIR / "analysis" / "plots"
DEMO_DATA_DIR = ROOT_DIR / "analysis" / "demo_data"

POLY_PRICE_COLUMNS = [
    "probability",
    "price",
    "lastPrice",
    "last_price",
    "yes_price",
    "yesPrice",
    "bestBid",
    "bid",
    "mid",
]
POLY_TIME_COLUMNS = [
    "timestamp",
    "time",
    "datetime",
    "createdAt",
    "updatedAt",
    "endDate",
    "closeTime",
]


def load_kraken_ohlc(csv_path: Path) -> pd.DataFrame:
    """Load Kraken OHLC candles into a DataFrame indexed by UTC timestamp."""

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        raise ValueError("Kraken CSV must include a 'timestamp' or 'datetime' column")

    df = df.sort_values("time").drop_duplicates(subset="time")
    df = df.set_index("time")

    numeric_columns = [col for col in ["open", "high", "low", "close", "vwap", "volume", "count"] if col in df.columns]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    return df


def _extract_outcome_price(value: object) -> float | None:
    """Attempt to parse an outcome price field that may be a list/JSON string."""

    if isinstance(value, (list, tuple)) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        try:
            parsed = literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and parsed:
            try:
                return float(parsed[0])
            except (TypeError, ValueError):
                return None
    return None


def _read_polymarket_csv(csv_path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Read a Polymarket CSV, falling back across common encodings.

    Windows-exported CSVs occasionally include smart quotes or other characters
    that are not UTF-8 friendly. Try the requested encoding first, then fall
    back through a few common options before surfacing a clear error.
    """

    encodings: list[str | None] = []
    if encoding:
        encodings.append(encoding)
    encodings.extend(["utf-8", "cp1252", "latin-1"])

    attempted: list[str] = []
    for enc in encodings:
        if enc in attempted:
            continue
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            attempted.append(enc)
            continue

    tried = ", ".join(enc for enc in attempted if enc)
    raise ValueError(
        f"Unable to decode {csv_path} using encodings: {tried}. "
        "Pass --polymarket-encoding to specify an explicit codec."
    )


def load_polymarket_markets(csv_path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Load Polymarket BTC market snapshots and return a time-indexed frame.

    The loader searches for a timestamp column and a probability/price column.
    If both best bid and best ask are available it will use their mid-price.
    """

    df = _read_polymarket_csv(csv_path, encoding=encoding)

    time_column = next((col for col in POLY_TIME_COLUMNS if col in df.columns), None)
    if time_column is None:
        raise ValueError("Polymarket CSV must include a timestamp-like column")

    if time_column == "timestamp":
        df["time"] = pd.to_datetime(df[time_column], unit="s", utc=True)
    else:
        df["time"] = pd.to_datetime(df[time_column], utc=True)

    best_bid = pd.to_numeric(df.get("bestBid"), errors="coerce") if "bestBid" in df.columns else None
    best_ask = pd.to_numeric(df.get("bestAsk"), errors="coerce") if "bestAsk" in df.columns else None

    probability_column = None
    if best_bid is not None and best_ask is not None:
        df["probability"] = (best_bid + best_ask) / 2
        probability_column = "probability"
    else:
        for candidate in POLY_PRICE_COLUMNS:
            if candidate in df.columns:
                series = pd.to_numeric(df[candidate], errors="coerce")
                if series.notna().any():
                    df["probability"] = series
                    probability_column = candidate
                    break

    if probability_column is None and "outcomePrices" in df.columns:
        df["probability"] = df["outcomePrices"].apply(_extract_outcome_price)
        if df["probability"].notna().any():
            probability_column = "outcomePrices"

    if probability_column is None:
        raise ValueError("Could not locate a probability/price column in Polymarket CSV")

    df = df.sort_values("time").drop_duplicates(subset="time")
    df = df.set_index("time")
    df["probability"] = df["probability"].astype(float)
    df["probability"] = df["probability"].ffill().bfill()

    return df[["probability"]]


def infer_interval_minutes(index: pd.DatetimeIndex) -> int:
    """Infer candle interval in minutes from a DatetimeIndex."""

    if len(index) < 2:
        return 60
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= 0:
        return 60
    return max(1, int(round(median_delta)))


def resample_polymarket(probabilities: pd.Series, target_index: pd.DatetimeIndex, interval_minutes: int) -> pd.Series:
    """Resample Polymarket series to the Kraken time grid with ffill/bfill."""

    frequency = f"{interval_minutes}min"
    resampled = probabilities.resample(frequency).ffill().bfill()

    # Align to Kraken timestamps; use nearest within one interval for robustness
    aligned = resampled.reindex(target_index, method="nearest", tolerance=pd.Timedelta(minutes=interval_minutes))
    if aligned.isna().any():
        aligned = aligned.ffill().bfill()
    return aligned


def _generate_demo_times(length: int, interval_minutes: int) -> pd.DatetimeIndex:
    """Create a UTC datetime index for synthetic data."""

    end_time = pd.Timestamp.utcnow().floor(f"{interval_minutes}min")
    start_time = end_time - pd.Timedelta(minutes=interval_minutes * (length - 1))
    return pd.date_range(start=start_time, periods=length, freq=f"{interval_minutes}min", tz="UTC")


def make_demo_data(length: int, interval_minutes: int, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic Kraken OHLC and Polymarket probability frames.

    Polymarket probabilities are nudged to *lead* BTC by a couple of intervals
    so the demo clearly demonstrates a lead/lag relationship.
    """

    rng = np.random.default_rng(seed)
    times = _generate_demo_times(length, interval_minutes)

    btc_returns = rng.normal(loc=0.0002, scale=0.0025, size=length)
    btc_close = pd.Series(btc_returns).add(1).cumprod() * 30000
    btc_open = btc_close.shift(1).fillna(btc_close.iloc[0] * (1 - btc_returns[0]))
    btc_high = pd.concat([btc_open, btc_close], axis=1).max(axis=1) * (1 + rng.normal(0.0002, 0.0005, size=length))
    btc_low = pd.concat([btc_open, btc_close], axis=1).min(axis=1) * (1 - rng.normal(0.0002, 0.0005, size=length))
    btc_volume = rng.uniform(10, 30, size=length)

    kraken_df = pd.DataFrame(
        {
            "timestamp": (times.view(np.int64) // 1_000_000_000),
            "open": btc_open.values,
            "high": btc_high.values,
            "low": btc_low.values,
            "close": btc_close.values,
            "volume": btc_volume,
            "count": rng.integers(10, 80, size=length),
        }
    )

    # Create Polymarket probabilities that drift with a small lead over BTC
    lead = 2
    shifted_returns = pd.Series(btc_returns).shift(-lead).fillna(0)
    poly_noise = rng.normal(loc=0, scale=0.0015, size=length)
    poly_changes = 0.15 * shifted_returns + poly_noise
    probabilities = pd.Series(0.6 + np.cumsum(poly_changes)).clip(0.05, 0.95)

    poly_df = pd.DataFrame(
        {
            "timestamp": (times.view(np.int64) // 1_000_000_000),
            "probability": probabilities,
            "bestBid": probabilities - 0.01,
            "bestAsk": probabilities + 0.01,
        }
    )

    return kraken_df, poly_df


def compute_correlations(
    btc_prices: pd.Series,
    polymarket_probs: pd.Series,
    interval_minutes: int,
    max_lag: int,
    rolling_windows_hours: Iterable[int],
) -> Tuple[pd.Series, dict, pd.DataFrame]:
    """Compute base, rolling, and lagged correlations."""

    btc_returns = btc_prices.pct_change().dropna()
    poly_changes = polymarket_probs.pct_change().dropna()
    combined = pd.concat([btc_returns.rename("btc_returns"), poly_changes.rename("poly_changes")], axis=1).dropna()

    base_correlation = combined["btc_returns"].corr(combined["poly_changes"])

    rolling = {}
    for hours in rolling_windows_hours:
        window = max(2, int(round((hours * 60) / interval_minutes)))
        rolling_corr = combined["btc_returns"].rolling(window=window, min_periods=2).corr(combined["poly_changes"])
        rolling[hours] = rolling_corr

    lags = range(-max_lag, max_lag + 1)
    lag_corrs: List[Tuple[int, float]] = []
    for lag in lags:
        if lag > 0:
            # Positive lag: Polymarket leads BTC by ``lag`` intervals
            shifted_poly = combined["poly_changes"].shift(lag)
            shifted_btc = combined["btc_returns"]
        elif lag < 0:
            # Negative lag: BTC leads Polymarket
            shifted_poly = combined["poly_changes"]
            shifted_btc = combined["btc_returns"].shift(-lag)
        else:
            shifted_poly = combined["poly_changes"]
            shifted_btc = combined["btc_returns"]
        if shifted_poly.count() == 0 or shifted_btc.count() == 0:
            corr_value = np.nan
        else:
            corr_value = shifted_btc.corr(shifted_poly)
        lag_corrs.append((lag, corr_value))

    lag_df = pd.DataFrame(lag_corrs, columns=["lag", "correlation"]).set_index("lag")

    return pd.Series({"pearson": base_correlation}), rolling, lag_df


def plot_price_overlay(
    btc_prices: pd.Series,
    polymarket_probs: pd.Series,
    output_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(btc_prices.index, btc_prices, label="BTC Close", color="tab:blue")
    ax1.set_ylabel("BTC Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(polymarket_probs.index, polymarket_probs, label="Polymarket Probability", color="tab:orange")
    ax2.set_ylabel("Probability", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("BTC Price vs Polymarket Probability")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rolling_correlation(rolling: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for hours, series in rolling.items():
        ax.plot(series.index, series, label=f"Rolling {hours}h")
    ax.set_title("Rolling Correlation (BTC returns vs Polymarket changes)")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Time")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_lag_correlation(lag_df: pd.DataFrame, interval_minutes: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lag_df.index, lag_df["correlation"], width=0.8, color="tab:green")
    ax.set_title("Lead/Lag Correlation (positive = Polymarket leads)")
    ax.set_xlabel(f"Lag (intervals of {interval_minutes} minutes)")
    ax.set_ylabel("Correlation")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kraken-csv", type=Path, default=DEFAULT_KRAKEN_PATH, help="Path to Kraken OHLC CSV")
    parser.add_argument(
        "--polymarket-csv",
        type=Path,
        default=DEFAULT_POLYMARKET_PATH,
        help="Path to Polymarket BTC markets CSV",
    )
    parser.add_argument(
        "--polymarket-encoding",
        type=str,
        default=None,
        help="Optional text encoding for Polymarket CSV (auto-tries utf-8, cp1252, latin-1)",
    )
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR, help="Directory for plot outputs")
    parser.add_argument("--max-lag", type=int, default=12, help="Lag window in intervals for cross-correlation")
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default="1,4",
        help="Comma-separated rolling window sizes in hours (e.g., '1,4,12')",
    )
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Generate synthetic demo CSVs (useful when real data is unavailable)",
    )
    parser.add_argument("--demo-length", type=int, default=240, help="Number of rows for demo data")
    parser.add_argument(
        "--demo-interval",
        type=int,
        default=60,
        help="Candle interval in minutes for demo data (default: 60)",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=DEMO_DATA_DIR,
        help="Output directory for generated demo CSVs",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively after saving")
    return parser.parse_args()


def run_visualization(args: argparse.Namespace) -> None:
    kraken_path: Path = args.kraken_csv
    polymarket_path: Path = args.polymarket_csv

    if args.demo_data:
        args.demo_dir.mkdir(parents=True, exist_ok=True)
        kraken_path = args.demo_dir / "demo_kraken.csv"
        polymarket_path = args.demo_dir / "demo_polymarket.csv"
        kraken_df, polymarket_df = make_demo_data(length=args.demo_length, interval_minutes=args.demo_interval)
        kraken_df.to_csv(kraken_path, index=False)
        polymarket_df.to_csv(polymarket_path, index=False)
        print(f"Generated demo Kraken data at {kraken_path}")
        print(f"Generated demo Polymarket data at {polymarket_path}")

    if not kraken_path.exists():
        raise FileNotFoundError(
            f"Kraken CSV not found: {kraken_path}. Pass --demo-data to generate synthetic data if needed."
        )
    if not polymarket_path.exists():
        raise FileNotFoundError(
            f"Polymarket CSV not found: {polymarket_path}. Pass --demo-data to generate synthetic data if needed."
        )

    kraken_df = load_kraken_ohlc(kraken_path)
    polymarket_df = load_polymarket_markets(polymarket_path, encoding=args.polymarket_encoding)

    interval_minutes = infer_interval_minutes(kraken_df.index)

    aligned_poly = resample_polymarket(polymarket_df["probability"], kraken_df.index, interval_minutes)
    pearson, rolling, lag_df = compute_correlations(
        btc_prices=kraken_df["close"],
        polymarket_probs=aligned_poly,
        interval_minutes=interval_minutes,
        max_lag=args.max_lag,
        rolling_windows_hours=[int(x) for x in args.rolling_windows.split(",") if x],
    )

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.plots_dir / "price_probability_overlay.png"
    rolling_path = args.plots_dir / "rolling_correlation.png"
    lag_path = args.plots_dir / "lag_correlation.png"

    plot_price_overlay(kraken_df["close"], aligned_poly, overlay_path)
    plot_rolling_correlation(rolling, rolling_path)
    plot_lag_correlation(lag_df, interval_minutes, lag_path)

    print("Correlation summary:")
    print(f"  Pearson correlation between returns: {pearson['pearson']:.4f}")
    for hours, series in rolling.items():
        latest = series.dropna().iloc[-1] if not series.dropna().empty else np.nan
        print(f"  Latest rolling {hours}h correlation: {latest:.4f}")

    best_lag = lag_df["correlation"].dropna().idxmax() if not lag_df["correlation"].dropna().empty else None
    if best_lag is not None:
        direction = "Polymarket leads" if best_lag > 0 else "BTC leads" if best_lag < 0 else "In sync"
        print(f"  Strongest lag correlation at {best_lag} intervals ({direction}) -> {lag_df.loc[best_lag, 'correlation']:.4f}")

    print(f"Plots saved to {args.plots_dir}")

    if args.show:
        plt.show()


def main() -> int:
    args = parse_args()
    run_visualization(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
