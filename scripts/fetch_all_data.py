"""Command-line script to fetch Kraken OHLCV data and Polymarket BTC markets."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Sequence

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.kraken_client import get_kraken_ohlc
from data.polymarket_client import filter_btc_markets, get_polymarket_markets

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Create an empty file with no headers to indicate the run happened
        path.touch()
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_and_save_kraken(
    pair: str, interval: int, since: int | None, output: Path, proxies: Dict[str, object] | None
) -> None:
    print(f"Fetching Kraken OHLCV for {pair} (interval {interval}m)...")
    candles = get_kraken_ohlc(pair=pair, interval=interval, since=since, proxies=proxies)
    write_csv(output, candles)
    print(f"Saved {len(candles)} rows to {output}")


def fetch_and_save_polymarket(
    limit: int,
    offset: int,
    closed: bool | None,
    proxies: Dict[str, object] | None,
    output: Path,
) -> None:
    print("Fetching Polymarket markets...")
    markets = get_polymarket_markets(limit=limit, offset=offset, closed=closed, proxies=proxies)
    btc_markets = filter_btc_markets(markets)
    print(f"Found {len(markets)} markets; {len(btc_markets)} matched Bitcoin keywords")
    write_csv(output, btc_markets)
    print(f"Saved {len(btc_markets)} BTC-related markets to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="XBTUSD", help="Kraken pair code (default: XBTUSD)")
    parser.add_argument("--interval", type=int, default=60, help="Kraken OHLC interval in minutes")
    parser.add_argument("--since", type=int, default=None, help="Unix timestamp to fetch data since (seconds)")
    parser.add_argument("--limit", type=int, default=500, help="Polymarket markets limit")
    parser.add_argument("--offset", type=int, default=0, help="Polymarket markets offset")
    parser.add_argument("--closed", type=str, choices=["true", "false"], default=None, help="Filter Polymarket markets by closed status")
    parser.add_argument(
        "--disable-proxies",
        action="store_true",
        help="Disable HTTP/HTTPS proxies for outbound requests (useful in restricted environments)",
    )
    parser.add_argument(
        "--kraken-output",
        type=Path,
        default=RAW_DATA_DIR / "kraken_btc_ohlc.csv",
        help="Output CSV path for Kraken data",
    )
    parser.add_argument(
        "--polymarket-output",
        type=Path,
        default=RAW_DATA_DIR / "polymarket_btc_markets.csv",
        help="Output CSV path for Polymarket data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closed_flag = None
    if args.closed is not None:
        closed_flag = args.closed.lower() == "true"

    proxies = {"http": None, "https": None} if args.disable_proxies else None

    try:
        fetch_and_save_kraken(
            pair=args.pair,
            interval=args.interval,
            since=args.since,
            output=args.kraken_output,
            proxies=proxies,
        )
    except (requests.HTTPError, requests.ConnectionError, RuntimeError) as exc:
        print(f"Failed to fetch Kraken data: {exc}")
        return 1

    try:
        fetch_and_save_polymarket(
            limit=args.limit,
            offset=args.offset,
            closed=closed_flag,
            proxies=proxies,
            output=args.polymarket_output,
        )
    except (requests.HTTPError, requests.ConnectionError, RuntimeError, ValueError) as exc:
        print(f"Failed to fetch Polymarket data: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
