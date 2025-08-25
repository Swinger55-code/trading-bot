# agent_discord.py
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

# --- your local modules ---
from notify import notify
from dl import onchain_snapshot, onchain_summary_for_ticker
from derivs import deriv_summary  # Coinglass-based derivatives summary (optional)

"""
Main agent for crypto trading alerts.

Scans a universe of symbols (CORE + CMC top 100) for breakout/retest (BUY)
and breakdown/retest (SELL) signals using EMAs/RSI/MACD/ATR + simple PA filters.
Optionally consults DeFiLlama (on-chain) and Coinglass (derivatives) for context,
and sends alerts to Discord via webhook.

Educational use only. Not financial advice.
"""

load_dotenv()

# -------------------------------
# Config / Environment
# -------------------------------

CMC_KEY: str = (os.getenv("CMC_API_KEY") or "").strip()
HEADERS_CMC = {"X-CMC_PRO_API_KEY": CMC_KEY}

# Core universe (comma-separated symbols in .env UNIVERSE; fallback shown)
CORE: List[str] = [s.strip().upper() for s in (os.getenv("UNIVERSE") or "ARB,ONDO,SEI,SUI").split(",") if s.strip()]

# Scan cadence & summaries
SCAN_INTERVAL_MIN: int = int(os.getenv("SCAN_INTERVAL_MIN") or 60)  # minutes
SUMMARY_HOURS: int = int(os.getenv("SUMMARY_HOURS") or 4)

# Map symbols -> DeFiLlama chain slug for on-chain gating
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

# Rate limiting / cooldowns
MAX_PER_HOUR = 3
MAX_PER_DAY = 10
PER_COIN_COOLDOWN_MIN = 90

STATE = {
    "h": 0,               # alerts this hour
    "d": 0,               # alerts today
    "last_h": None,       # last hour reset
    "last_d": None,       # last day reset
    "per_coin": {},       # sym -> last alert time
}


# -------------------------------
# Helpers
# -------------------------------

def now() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def load_universe() -> List[str]:
    """CORE + CMC top 100 (if key present)."""
    syms = set(CORE)
    if not CMC_KEY:
        print("[warn] No CMC_API_KEY, scanning only CORE.")
        return sorted(syms)
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        r = requests.get(url, headers=HEADERS_CMC, params={"limit": 100, "convert": "USD"}, timeout=30)
        r.raise_for_status()
        for coin in r.json().get("data", []):
            s = coin.get("symbol")
            if s:
                syms.add(s.upper())
    except Exception as e:
        print("[cmc] listings error:", e)
    return sorted(syms)


def cmc_ohlcv_daily(symbol: str, days: int = 220) -> Optional[pd.DataFrame]:
    """Daily OHLCV from CoinMarketCap; None on failure."""
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
    """(total_mcap_usd, btc_dominance) or (None, None)."""
    try:
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        r = requests.get(url, headers=HEADERS_CMC, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        return d["quote"]["USD"]["total_market_cap"], d["btc_dominance"]
    except Exception:
        return None, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMAs, RSI, MACD(+hist), ATR, swing HL(20), vol MA(20)."""
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["swing_high"] = df["high"].rolling(20).max()
    df["swing_low"] = df["low"].rolling(20).min()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


def fib_levels(df: pd.DataFrame, lookback: int = 120) -> dict[str, float]:
    recent = df.tail(lookback)
    hi = recent["high"].max()
    lo = recent["low"].min()
    rng = max(hi - lo, 1e-9)
    return {"0.618": hi - 0.618 * rng, "0.5": lo + 0.5 * rng, "0.382": hi - 0.382 * rng}


# -------------------------------
# Signal logic
# -------------------------------

def breakout_retest_signal(sym: str, df: pd.DataFrame) -> Optional[str]:
    """Breakout + retest long signal with optional on-chain gate."""
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

    # Soft on-chain confirmation via snapshot
    chain = CHAIN.get(sym)
    ok_on = True
    if chain:
        snap = onchain_snapshot(chain)
        score = 0
        if snap["tvl"] is not None and snap["tvl"] >= 5:
            score += 1
        for key, thr in [("dex_vol", 5_000_000), ("perps_vol", 5_000_000), ("fees", 200_000), ("stable_mcap", 100_000_000)]:
            v = snap[key]
            if isinstance(v, (int, float)) and v > thr:
                score += 1
        ok_on = score >= 1

    if all([volume_ok, ema_trend, momentum, above_fib, broke_high, retest_ok, ok_on]):
        atr = last["atr"]
        stop = last["close"] - 1.5 * atr
        rr = max(last["close"] - stop, 1e-6)
        tp1 = last["close"] + 2 * rr
        tp2 = last["close"] + 3 * rr

        # Context lines
        oc = onchain_summary_for_ticker(sym) or "none"
        deriv = deriv_summary(sym, last["close"])
        deriv_txt = f" | Derivs: {deriv}" if deriv else ""

        return (
            f"[BUY] {sym} @ {last['close']:.6f} | Lvg 3–5x | "
            f"SL {stop:.6f} | TP {tp1:.6f}/{tp2:.6f} | "
            f"RSI {last['rsi']:.1f} MACDh {last['macd_hist']:.4f} | On-chain: {oc}{deriv_txt}"
        )

    return None


def exit_signal(sym: str, df: pd.DataFrame) -> Optional[str]:
    """Breakdown + retest short signal (inverse)."""
    if df is None or len(df) < 200:
        return None

    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    fib = fib_levels(df, 120)

    # Inverse checks
    volume_ok = last["volume"] > 1.2 * (df["vol_ma"].iloc[-2] or 1)
    ema_trend = last["close"] < last["ema50"] < last["ema200"]
    momentum = last["macd_hist"] < 0 and last["rsi"] < 50
    below_fib = last["close"] < fib["0.382"]
    broke_low = last["close"] < df["swing_low"].iloc[-2] and prev["close"] >= df["swing_low"].iloc[-3]
    retest_ok = last["high"] >= df["swing_low"].iloc[-2] * 0.995 and last["close"] <= df["swing_low"].iloc[-2]

    if all([volume_ok, ema_trend, momentum, below_fib, broke_low, retest_ok]):
        atr = last["atr"]
        stop = last["close"] + 1.5 * atr
        rr = max(stop - last["close"], 1e-6)
        tp1 = last["close"] - 2 * rr
        tp2 = last["close"] - 3 * rr

        oc = onchain_summary_for_ticker(sym) or "none"
        deriv = deriv_summary(sym, last["close"])
        deriv_txt = f" | Derivs: {deriv}" if deriv else ""

        return (
            f"[SELL] {sym} @ {last['close']:.6f} | Lvg 3–5x | "
            f"SL {stop:.6f} | TP {tp1:.6f}/{tp2:.6f} | "
            f"RSI {last['rsi']:.1f} MACDh {last['macd_hist']:.4f} | On-chain: {oc}{deriv_txt}"
        )

    return None


# -------------------------------
# Rate limiting
# -------------------------------

def can_send(sym: str) -> bool:
    n = now()
    # hourly reset
    if not STATE["last_h"] or (n - STATE["last_h"]).seconds >= 3600:
        STATE.update({"h": 0, "last_h": n})
    # daily reset
    if not STATE["last_d"] or (n - STATE["last_d"]).days >= 1:
        STATE.update({"d": 0, "last_d": n, "per_coin": {}})
    # global limits
    if STATE["h"] >= MAX_PER_HOUR or STATE["d"] >= MAX_PER_DAY:
        return False
    # per-coin cooldown
    last_alert = STATE["per_coin"].get(sym, datetime(1970, 1, 1, tzinfo=timezone.utc))
    return (n - last_alert).seconds >= PER_COIN_COOLDOWN_MIN * 60


def mark(sym: str) -> None:
    STATE["h"] += 1
    STATE["d"] += 1
    STATE["per_coin"][sym] = now()


# -------------------------------
# Scan loop
# -------------------------------

def scan_once() -> List[str]:
    picks: List[str] = []
    for sym in load_universe():
        try:
            df = cmc_ohlcv_daily(sym, days=220)
            # BUY
            sig = breakout_retest_signal(sym, df)
            if sig and can_send(sym):
                print(sig)
                notify(sig)
                picks.append(sig)
                mark(sym)

            # SELL
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
    uni = load_universe()
    print("Agent live. Scanning:", ", ".join(uni[:15]), "...")
    total, dom = cmc_global()
    if total and dom:
        print(f"[macro] Total mcap=${total:,.0f} | BTC dom={dom:.1f}%")

    last_sum = 0.0
    while True:
        scan_once()
        if time.time() - last_sum > SUMMARY_HOURS * 3600:
            notify(f"[summary] agent running; {SUMMARY_HOURS}h checkpoint.")
            last_sum = time.time()
        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    if not CMC_KEY:
        print("[fatal] Missing CMC_API_KEY in .env")
    else:
        main()

