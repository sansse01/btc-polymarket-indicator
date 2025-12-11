"""Convenience exports for data fetching clients.

The actual implementation now lives in the `data` package. Import functions
from here for backward compatibility.
"""

from data.kraken_client import get_kraken_ohlc
from data.polymarket_client import (
    filter_btc_markets,
    get_all_polymarket_markets,
    get_polymarket_markets,
)

__all__ = [
    "get_kraken_ohlc",
    "get_polymarket_markets",
    "get_all_polymarket_markets",
    "filter_btc_markets",
]
