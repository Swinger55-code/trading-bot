"""
Helpers for interacting with DefiLlama APIs.

This module encapsulates a handful of calls to the public DefiLlama API
to fetch total value locked (TVL) and on‑chain activity metrics for
various chains. These metrics are used to provide soft on‑chain
confirmation for trading signals.

The API does not require an API key and is rate‑limited on the server
side. If you need more reliability, consider caching responses or
throttling requests.
"""

import requests
import pandas as pd
from typing import Optional

# Base URL for DefiLlama API
BASE = "https://api.llama.fi"


def _get(url: str, **params) -> dict:
    """Perform a GET request and return parsed JSON.

    Raises:
        HTTPError if the request fails.
    """
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def chain_tvl_df(chain: str) -> pd.DataFrame:
    """Historical TVL for a chain (daily).

    Returns:
        A pandas DataFrame indexed by date with a ``tvl`` column.
    """
    js = _get(f"{BASE}/v2/historicalChainTvl/{chain}")
    df = pd.DataFrame(js)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df.set_index("date", inplace=True)
    return df


def _pct_change_last_24(series: pd.Series) -> Optional[float]:
    """Return the percent change between the last two values in the series.
    Returns ``None`` if there aren't at least two observations or the
    previous value is zero.
    """
    if len(series) < 2:
        return None
    a, b = series.iloc[-1], series.iloc[-2]
    if not b:
        return None
    return (a - b) / b * 100.0


def onchain_snapshot(chain: str) -> dict:
    """Light snapshot used to increase confidence on breakouts.

    This function combines several API calls from DefiLlama to return
    a snapshot dictionary with the following keys:

    - ``tvl``: percent change in TVL over the last 24h (float or None)
    - ``dex_vol``: 24‑hour DEX volume in USD (float or None)
    - ``perps_vol``: 24‑hour perpetuals volume in USD (float or None)
    - ``fees``: total fees generated in the last 24h (float or None)
    - ``revenue``: total revenue in the last 24h (float or None)
    - ``stable_mcap``: market cap of stables issued on the chain (float or None)

    Missing values are represented as ``None``. If an API call fails,
    that metric will be ``None``.
    """
    snap = {
        "tvl": None,
        "dex_vol": None,
        "perps_vol": None,
        "fees": None,
        "revenue": None,
        "stable_mcap": None,
    }
    # TVL percent change
    try:
        tvl = chain_tvl_df(chain)
        if "tvl" in tvl:
            snap["tvl"] = _pct_change_last_24(tvl["tvl"])
    except Exception:
        pass
    # DEX volume
    try:
        js = _get(f"{BASE}/summary/dexs/{chain}")
        snap["dex_vol"] = js.get("data", {}).get("total24h")
    except Exception:
        pass
    # Perpetuals volume
    try:
        js = _get(f"{BASE}/summary/perps/{chain}")
        snap["perps_vol"] = js.get("data", {}).get("total24h")
    except Exception:
        pass
    # Fees and revenue
    try:
        js = _get(f"{BASE}/summary/fees/{chain}")
        snap["fees"] = js.get("total24hFees")
        snap["revenue"] = js.get("total24hRevenue")
    except Exception:
        pass
    # Stablecoin market cap
    try:
        j = requests.get("https://stablecoins.llama.fi/stablecoinchains", timeout=30).json()
        for row in j.get("chains", []):
            if row.get("chain", "").lower() == chain.lower():
                snap["stable_mcap"] = row.get("total", {}).get("totalCirculatingUSD")
                break
    except Exception:
        pass
    return snap