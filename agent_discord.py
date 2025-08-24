"""
Main agent for crypto trading alerts.

This script performs periodic scans of the top 100 cryptocurrencies on
CoinMarketCap (plus any user‑defined universe) to detect breakout and
retest patterns that may warrant a leveraged long entry. It uses a
combination of trend (moving averages), momentum (RSI/MACD), volume
expansion and price action (swing high breakout with retest) to
generate buy signals. Optionally, it consults DefiLlama to confirm
on‑chain activity is healthy. It also generates corresponding sell
signals when bearish conditions arise (trend reverses, momentum turns
negative and price breaks below support).

Alerts are delivered to Discord via a webhook specified in ``.env``.
Rate limiting ensures no more than a handful of alerts are sent per
hour/day and that each coin is only alerted once every few hours.

This bot is intended for educational purposes and is **not
financial advice**. Use at your own risk.
"""

from __future__ import annotations

import os
import time
import requests
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

from notify import notify
from dl import onchain_snapshot

load_dotenv()

# CoinMarketCap API key and headers
CMC_KEY: str = (os.getenv("CMC_API_KEY") or "").strip()
HEADERS_CMC = {"X-CMC_PRO_API_KEY": CMC_KEY}

# Core universe of symbols to monitor (always uppercase)
CORE: List[str] = [s.strip().upper() for s in (os.getenv("UNIVERSE") or "ARB,ONDO,SEI,SUI").split(",") if s.strip()]

# Tuning parameters
SCAN_INTERVAL_MIN: int = int(os.getenv("SCAN_INTERVAL_MIN") or 60)
SUMMARY_HOURS: int = int(os.getenv("SUMMARY_HOURS") or 4)

# Mapping from symbol to DefiLlama chain name for on‑chain confirmation
CHAIN: dict[str, str] = {
    "ARB": "arbitrum",
    "ONDO": "ethereum",
    "SEI": "sei",
    "SUI": "sui",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "TON": "ton",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "LINK": "ethereum",
    "OP": "optimism",
}

# Simple rate limiting / cool‑down for alerts
MAX_PER_HOUR = 3
MAX_PER_DAY = 10
PER_COIN_COOLDOWN_MIN = 90

# Internal state for rate limiting
STATE = {
    "h": 0,  # number of alerts sent in the current hour
    "d": 0,  # number of alerts sent today
    "last_h": None,  # timestamp of last hour reset
    "last_d": None,  # timestamp of last day reset
    "per_coin": {},  # last alert time per coin
}


def now() -> datetime:
    """Return current UTC time with timezone information."""
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def load_universe() -> List[str]:
    """Your core coins plus CMC top 100 by market cap.

    Returns a sorted list of uppercase symbols. If the CoinMarketCap
    API key is missing or an error occurs, only the core universe is
    returned.
    """
    syms = set(CORE)
    if not CMC_KEY:
        print("[warn] No CMC_API_KEY, scanning only CORE.")
        return sorted(syms)
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        r = requests.get(url, headers=HEADERS_CMC, params={"limit": 100, "convert": "USD"}, timeout=30)
        r.raise_for_status()
        for coin in r.json().get("data", []):
            syms.add(coin["symbol"].upper())
    except Exception as e:
        print("[cmc] listings error:", e)
    return sorted(syms)


def cmc_ohlcv_daily(symbol: str, days: int = 220) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data for a symbol from CoinMarketCap.

    Returns a DataFrame with columns [open, high, low, close, volume] indexed
    by timestamp. Returns None if the request fails or no data is available.
    """
    if not CMC_KEY:
        return None
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    try:
        r = requests.get(
            url,
            headers=HEADERS_CMC,
            params={"symbol": symbol, "convert": "USD", "count": days, "interval": "1d"},
            timeout=40,
        )
        if r.status_code != 200:
            return None
        q = r.json().get("data", {}).get("quotes", [])
        if not q:
            return None
        df = pd.DataFrame(
            [
                {
                    "time": pd.to_datetime(x["time_open"]),
                    "open": x["quote"]["USD"]["open"],
                    "high": x["quote"]["USD"]["high"],
                    "low": x["quote"]["USD"]["low"],
                    "close": x["quote"]["USD"]["close"],
                    "volume": x["quote"]["USD"]["volume"],
                }
                for x in q
            ]
        ).set_index("time")
        return df
    except Exception as e:
        print("[cmc] ohlcv error", symbol, e)
        return None


def cmc_global() -> tuple[Optional[float], Optional[float]]:
    """Return total crypto market cap and BTC dominance.

    Returns (total_mcap_usd, btc_dominance) or (None, None) on failure.
    """
    try:
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        r = requests.get(url, headers=HEADERS_CMC, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        return d["quote"]["USD"]["total_market_cap"], d["btc_dominance"]
    except Exception:
        return None, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append technical indicators to the DataFrame.

    Includes EMAs (20/50/200), RSI, MACD and histogram, ATR, swing highs/lows,
    and volume moving average. Returns a copy of the DataFrame with new
    columns added.
    """
    df = df.copy()
    # Exponential moving averages
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    # MACD
    macd = MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    # ATR
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    # Swing highs/lows over the last 20 days
    df["swing_high"] = df["high"].rolling(20).max()
    df["swing_low"] = df["low"].rolling(20).min()
    # Volume moving average
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


def fib_levels(df: pd.DataFrame, lookback: int = 120) -> dict[str, float]:
    """Return basic Fibonacci retracement levels for the lookback period."""
    recent = df.tail(lookback)
    hi = recent["high"].max()
    lo = recent["low"].min()
    rng = max(hi - lo, 1e-9)
    return {
        "0.618": hi - 0.618 * rng,
        "0.5": lo + 0.5 * rng,
        "0.382": hi - 0.382 * rng,
    }


def breakout_retest_signal(sym: str, df: pd.DataFrame) -> Optional[str]:
    """Determine if the symbol exhibits a breakout + retest buy signal.

    Conditions:

      • Trend: price > EMA50 > EMA200
      • Momentum: RSI > 50 and MACD histogram > 0
      • Breakout: close above prior 20‑day swing high with volume expansion
      • Retest: intraday low tags the old swing high (~+0.5%) and closes back above it
      • On‑chain confirmation (soft): at least one on‑chain metric shows meaningful uptick

    Returns a formatted alert string or ``None`` if no signal.
    """
    if df is None or len(df) < 200:
        return None
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    fib = fib_levels(df, 120)
    # Technical checks
    volume_ok = last["volume"] > 1.2 * (df["vol_ma"].iloc[-2] or 1)
    ema_trend = last["close"] > last["ema50"] > last["ema200"]
    momentum = last["macd_hist"] > 0 and last["rsi"] > 50
    above_fib = last["close"] > fib["0.618"]
    broke_high = last["close"] > df["swing_high"].iloc[-2] and prev["close"] <= df["swing_high"].iloc[-3]
    retest_ok = last["low"] <= df["swing_high"].iloc[-2] * 1.005 and last["close"] >= df["swing_high"].iloc[-2]
    # On‑chain confirmation (soft)
    chain = CHAIN.get(sym)
    ok_on = True
    on_txt = " | On-chain: none"
    if chain:
        snap = onchain_snapshot(chain)
        parts: List[str] = []
        score = 0
        # TVL percent change ≥ +5%
        if snap["tvl"] is not None and snap["tvl"] >= 5:
            score += 1
            parts.append(f"TVL {snap['tvl']:+.1f}%")
        # Additional metrics: volumes, fees, stable mcap
        for key, label, thr in [
            ("dex_vol", "DEX", 5_000_000),
            ("perps_vol", "PERPS", 5_000_000),
            ("fees", "FEES", 200_000),
            ("stable_mcap", "STABLES", 100_000_000),
        ]:
            v = snap[key]
            if isinstance(v, (int, float)) and v > thr:
                score += 1
                parts.append(f"{label} ${v:,.0f}")
        ok_on = score >= 1
        on_txt = " | On-chain: " + (" + ".join(parts) if parts else "none")
    if all([volume_ok, ema_trend, momentum, above_fib, broke_high, retest_ok, ok_on]):
        atr = last["atr"]
        # Determine stop loss and take profits based off ATR; aligns with 3–5× leverage risk
        stop = last["close"] - 1.5 * atr
        rr = max(last["close"] - stop, 1e-6)
        tp1 = last["close"] + 2 * rr
        tp2 = last["close"] + 3 * rr
        return (
            f"[BUY] {sym} @ {last['close']:.6f} | Lvg 3–5x | SL {stop:.6f} | TP {tp1:.6f}/{tp2:.6f} "
            f"| RSI {last['rsi']:.1f} MACDh {last['macd_hist']:.4f}{on_txt}"
        )
    return None


def exit_signal(sym: str, df: pd.DataFrame) -> Optional[str]:
    """Determine if the symbol exhibits a bearish breakdown + retest sell signal.

    Conditions mirror the long signal but in reverse:

      • Trend: price < EMA50 < EMA200
      • Momentum: RSI < 50 and MACD histogram < 0
      • Breakdown: close below prior 20‑day swing low with volume expansion
      • Retest: intraday high tags the old swing low (~−0.5%) and closes back below it
      • Optional: price below lower Fibonacci level (0.382) to avoid shorting into support

    Returns a formatted alert string or ``None`` if no signal.
    """
    if df is None or len(df) < 200:
        return None
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    fib = fib_levels(df, 120)
    # Technical checks (inverse of breakout)
    volume_ok = last["volume"] > 1.2 * (df["vol_ma"].iloc[-2] or 1)
    ema_trend = last["close"] < last["ema50"] < last["ema200"]
    momentum = last["macd_hist"] < 0 and last["rsi"] < 50
    below_fib = last["close"] < fib["0.382"]
    broke_low = last["close"] < df["swing_low"].iloc[-2] and prev["close"] >= df["swing_low"].iloc[-3]
    retest_ok = last["high"] >= df["swing_low"].iloc[-2] * 0.995 and last["close"] <= df["swing_low"].iloc[-2]
    if all([volume_ok, ema_trend, momentum, below_fib, broke_low, retest_ok]):
        atr = last["atr"]
        # Set stop above current price and profit targets below; scale with leverage
        stop = last["close"] + 1.5 * atr
        rr = max(stop - last["close"], 1e-6)
        tp1 = last["close"] - 2 * rr
        tp2 = last["close"] - 3 * rr
        return (
            f"[SELL] {sym} @ {last['close']:.6f} | Lvg 3–5x | SL {stop:.6f} | TP {tp1:.6f}/{tp2:.6f} "
            f"| RSI {last['rsi']:.1f} MACDh {last['macd_hist']:.4f}"
        )
    return None


def can_send(sym: str) -> bool:
    """Return True if we can send an alert for this symbol now.

    Implements hourly and daily rate limits and per‑coin cool‑down.
    """
    n = now()
    # Reset hourly counter
    if not STATE["last_h"] or (n - STATE["last_h"]).seconds >= 3600:
        STATE.update({"h": 0, "last_h": n})
    # Reset daily counter
    if not STATE["last_d"] or (n - STATE["last_d"]).days >= 1:
        STATE.update({"d": 0, "last_d": n, "per_coin": {}})
    # Check global limits
    if STATE["h"] >= MAX_PER_HOUR or STATE["d"] >= MAX_PER_DAY:
        return False
    # Check per‑coin cool‑down
    last_alert = STATE["per_coin"].get(sym, datetime(1970, 1, 1, tzinfo=timezone.utc))
    return (n - last_alert).seconds >= PER_COIN_COOLDOWN_MIN * 60


def mark(sym: str) -> None:
    """Record that we sent an alert for this symbol now."""
    STATE["h"] += 1
    STATE["d"] += 1
    STATE["per_coin"][sym] = now()


def scan_once() -> List[str]:
    """Perform a single scan across the universe and return any alerts."""
    picks: List[str] = []
    for sym in load_universe():
        try:
            df = cmc_ohlcv_daily(sym, days=220)
            # Check for buy signal
            sig = breakout_retest_signal(sym, df)
            if sig and can_send(sym):
                print(sig)
                notify(sig)
                picks.append(sig)
                mark(sym)
            # Check for sell signal
            sig_sell = exit_signal(sym, df)
            if sig_sell and can_send(sym):
                print(sig_sell)
                notify(sig_sell)
                picks.append(sig_sell)
                mark(sym)
        except Exception as e:
            print("ERR", sym, e)
    if not picks:
        print("[scan] no high-confidence signals")
    return picks


def main() -> None:
    """Main loop for the agent."""
    uni = load_universe()
    print("Agent live. Scanning:", ", ".join(uni[:15]), "...")
    total, dom = cmc_global()
    if total and dom:
        print(f"[macro] Total mcap=${total:,.0f} | BTC dom={dom:.1f}%")
    last_sum = 0.0
    while True:
        scan_once()
        # Send periodic summary messages
        if time.time() - last_sum > SUMMARY_HOURS * 3600:
            notify(f"[summary] agent running; {SUMMARY_HOURS}h checkpoint.")
            last_sum = time.time()
        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    if not CMC_KEY:
        print("[fatal] Missing CMC_API_KEY in .env")
    else:
        main()