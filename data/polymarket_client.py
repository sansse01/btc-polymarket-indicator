"""Client utilities for interacting with Polymarket's Gamma API."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import requests

POLYMARKET_BASE_URL = "https://gamma-api.polymarket.com"


def get_polymarket_markets(
    limit: int = 100,
    offset: int = 0,
    closed: Optional[bool] = None,
    proxies: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict[str, object]]:
    """Fetch markets from Polymarket Gamma API.

    Args:
        limit: Maximum number of markets to return.
        offset: Pagination offset.
        closed: Filter by market closed status when provided.
        proxies: Optional requests-compatible proxy mapping to bypass restricted environments.

    Returns:
        A list of market dictionaries as returned by the API.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        ValueError: If the response body is not a recognized structure.
    """

    params: Dict[str, object] = {"limit": limit, "offset": offset}
    if closed is not None:
        params["closed"] = str(closed).lower()

    response = requests.get(
        f"{POLYMARKET_BASE_URL}/markets", params=params, timeout=30, proxies=proxies
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], Sequence):
            return list(payload["data"])
        if "markets" in payload and isinstance(payload["markets"], Sequence):
            return list(payload["markets"])

    raise ValueError("Unexpected Polymarket response structure")


def filter_btc_markets(markets: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Filter markets to only those related to Bitcoin.

    This performs a case-insensitive keyword search over the question/name/ticker
    fields to identify Bitcoin-oriented markets.
    """

    keywords = ("btc", "bitcoin", "satoshi", "sats")
    btc_markets: List[Dict[str, object]] = []

    for market in markets:
        text_fields = [
            str(market.get("question", "")),
            str(market.get("name", "")),
            str(market.get("ticker", "")),
        ]
        haystack = " ".join(text_fields).lower()
        if any(keyword in haystack for keyword in keywords):
            btc_markets.append(market)

    return btc_markets
