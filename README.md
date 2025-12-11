# btc-polymarket-indicator
Market indicator that tracks Polymarket prediction odds correlation to eventual BTC movement.

## Setup

1. Create a Python 3.12+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project layout

- `scripts/fetch_all_data.py`: CLI that downloads Kraken OHLCV candles plus Polymarket BTC-related markets into CSV files under `data/raw/`.
- `data/kraken_client.py` and `data/polymarket_client.py`: API helpers for Kraken and Polymarket (re-exported via `fetch_data.py` for backward compatibility).
- `analysis/correlation_visualizer.py`: Loads CSV snapshots, aligns timestamps, computes correlations, and produces plots.

## Fetching data

Run the helper script to pull fresh datasets:

```bash
python scripts/fetch_all_data.py --pair XBTUSD --interval 60 --limit 500 --pages 4
```

Options of note:

- `--disable-proxies` to bypass any configured HTTP/HTTPS proxies.
- `--kraken-output` and `--polymarket-output` to change where CSVs are written.
- `--closed true|false` to filter Polymarket markets by closed status.

The script prints progress to stdout and writes CSV files even when no rows are returned (empty files signal the attempt).

## Analyzing correlations

Once you have CSVs, generate plots and summary stats with:

```bash
python analysis/correlation_visualizer.py --kraken-csv data/raw/kraken_btc_ohlc.csv --polymarket-csv data/raw/polymarket_btc_markets.csv
```

Flags of interest:

- `--plots-dir`: Where to save the overlay, rolling correlation, and lead/lag bar plots (default: `analysis/plots/`).
- `--rolling-windows`: Comma-separated window sizes (hours) for rolling correlations (default: `1,4`).
- `--max-lag`: Number of intervals to test in each direction for lead/lag analysis (default: 12 intervals).
- `--show`: Open the plots interactively after saving them.

No data yet? Use the built-in demo generator:

```bash
python analysis/correlation_visualizer.py --demo-data --demo-length 240 --demo-interval 60
```

This writes synthetic Kraken and Polymarket CSVs into `analysis/demo_data/` and produces the same plots, which are handy for validating the workflow before hitting live APIs.
