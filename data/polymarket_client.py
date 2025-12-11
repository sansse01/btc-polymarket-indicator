"""Client utilities for interacting with Polymarket's Gamma API."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

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


def get_all_polymarket_markets(
    *,
    limit: int = 500,
    offset: int = 0,
    closed: Optional[bool] = None,
    proxies: Optional[Dict[str, Optional[str]]] = None,
    max_pages: Optional[int] = None,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, object]]:
    """Fetch multiple pages of Polymarket markets until exhausted.

    Args:
        limit: Page size for each request (Gamma API caps at 500).
        offset: Initial offset for pagination (useful for resuming).
        closed: Filter by closed status when provided.
        proxies: Optional requests-compatible proxy mapping.
        max_pages: Optional safety cap on the number of pages to fetch.
        on_page: Optional callback invoked as ``on_page(page_index, batch_size)``
            after each fetch, useful for progress logging.

    Returns:
        A combined list of market dictionaries across all fetched pages.
    """

    all_markets: List[Dict[str, object]] = []
    page = 0
    current_offset = offset

    while True:
        batch = get_polymarket_markets(
            limit=limit, offset=current_offset, closed=closed, proxies=proxies
        )
        if on_page:
            on_page(page, len(batch))
        if not batch:
            break
        all_markets.extend(batch)
        page += 1
        if max_pages is not None and page >= max_pages:
            break
        current_offset += limit

    return all_markets


def _gather_text_fields(market: Dict[str, object]) -> str:
    fields = [
        str(market.get("question", "")),
        str(market.get("name", "")),
        str(market.get("ticker", "")),
        str(market.get("slug", "")),
        str(market.get("category", "")),
        str(market.get("description", "")),
        str(market.get("resolutionSource", "")),
        str(market.get("cgAssetName", "")),
    ]

    events = market.get("events")
    if isinstance(events, Sequence):
        for event in events:
            if isinstance(event, dict):
                for key in ("title", "slug", "ticker", "description"):
                    fields.append(str(event.get(key, "")))

    series = market.get("series")
    if isinstance(series, Sequence):
        for entry in series:
            if isinstance(entry, dict):
                for key in ("title", "subtitle", "slug", "ticker", "cgAssetName"):
                    fields.append(str(entry.get(key, "")))

    outcomes = market.get("outcomes")
    if isinstance(outcomes, Sequence):
        for outcome in outcomes:
            fields.append(str(outcome))

    return " ".join(fields).lower()


def filter_btc_markets(markets: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Filter markets to only those related to Bitcoin.

    This performs a case-insensitive keyword search over the question/name/ticker
    fields to identify Bitcoin-oriented markets.
    """

    keywords = ("btc", "bitcoin", "satoshi", "sats")
    btc_markets: List[Dict[str, object]] = []

    for market in markets:
        haystack = _gather_text_fields(market)
        if any(keyword in haystack for keyword in keywords):
            btc_markets.append(market)

    return btc_markets
