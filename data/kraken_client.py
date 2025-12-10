"""Client utilities for fetching OHLCV data from the Kraken public API."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import requests

KRAKEN_BASE_URL = "https://api.kraken.com/0/public"


def get_kraken_ohlc(
    pair: str,
    interval: int = 1,
    since: Optional[int] = None,
    proxies: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict[str, object]]:
    """Fetch OHLCV candles for the requested trading pair.

    Args:
        pair: Trading pair code (e.g., "XBTUSD", "XXBTZUSD").
        interval: Candle interval in minutes. Kraken supports specific values.
        since: Return data since this Unix timestamp (in seconds). Optional.
        proxies: Optional requests-compatible proxy mapping to bypass restricted environments.

    Returns:
        A list of dictionaries containing OHLCV data with UTC datetime strings.

    Raises:
        RuntimeError: If the Kraken API reports an error in the payload.
        requests.HTTPError: If the HTTP request fails.
    """

    params: Dict[str, object] = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since

    response = requests.get(
        f"{KRAKEN_BASE_URL}/OHLC", params=params, timeout=30, proxies=proxies
    )
    response.raise_for_status()

    payload = response.json()
    api_errors = payload.get("error")
    if api_errors:
        raise RuntimeError(f"Kraken API error(s): {api_errors}")

    result = payload.get("result", {})
    pair_key = next((key for key in result.keys() if key != "last"), None)
    if pair_key is None or pair_key not in result:
        raise RuntimeError("Unexpected Kraken response structure; pair data missing")

    ohlc_rows = []
    for entry in result[pair_key]:
        if len(entry) < 8:
            # Skip malformed entries instead of failing the entire request
            continue
        timestamp, open_, high, low, close, vwap, volume, count = entry[:8]
        ohlc_rows.append(
            {
                "timestamp": int(timestamp),
                "datetime": datetime.utcfromtimestamp(int(timestamp)).isoformat(),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "vwap": float(vwap),
                "volume": float(volume),
                "count": int(count),
                "pair": pair,
            }
        )

    return ohlc_rows
