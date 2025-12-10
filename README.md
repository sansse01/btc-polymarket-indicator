# btc-polymarket-indicator
Market indicator that tracks polymarket prediction odds correlation to eventual BTC movement

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Verify Python 3.12+ is available (the scripts are developed against 3.12 in this environment).

## Fetching data

The project ships a helper script that pulls Kraken OHLCV candles and Polymarket markets, then writes CSV files into `data/raw/`.

```bash
python scripts/fetch_all_data.py --pair XBTUSD --interval 60 --limit 500
```

Options of note:

- `--disable-proxies` to bypass any configured HTTP/HTTPS proxies.
- `--kraken-output` and `--polymarket-output` to change where CSVs are written.
- `--closed true|false` to filter Polymarket markets by closed status.
