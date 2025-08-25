# derivs.py — Coinglass-driven derivatives summary for alerts
import os, requests
from typing import Optional, Tuple

KEY  = os.getenv("COINGLASS_API_KEY")
BASE = os.getenv("COINGLASS_BASE", "https://open-api.coinglass.com/public/v2")
HDRS = {"CG-API-KEY": KEY} if KEY else {}

FUND_HI = float(os.getenv("DERIV_FUNDING_HI", 0.0015))  # 0.15%/8h
FUND_LO = float(os.getenv("DERIV_FUNDING_LO", 0.0008))  # 0.08%/8h
OI_1H_T  = float(os.getenv("DERIV_OI_1H", 3))           # 3%

def _sym(sym: str) -> str:
    # adapt if your account uses a different symbol format
    return f"{sym.upper()}USDT"

def _get(path: str, params: dict) -> Optional[dict]:
    if not KEY:
        return None
    try:
        r = requests.get(f"{BASE}{path}", params=params, headers=HDRS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_funding(sym: str) -> Optional[float]:
    """Return latest funding rate (fraction per 8h), e.g. 0.0008 = 0.08%/8h."""
    s = _sym(sym)
    # NOTE: Replace path/keys with your Coinglass plan’s endpoint spec.
    # Example shape:
    data = _get("/perpetual/fundingRate", {"symbol": s})
    try:
        if data and data.get("data"):
            # pick the most recent item; adjust key names as per response
            row = data["data"][0]
            return float(row.get("rate") or row.get("fundingRate"))
    except Exception:
        pass
    return None

def fetch_oi_1h_change(sym: str) -> Optional[float]:
    """Return Open Interest % change over the last hour."""
    s = _sym(sym)
    data = _get("/openInterest", {"symbol": s, "interval": "1h"})
    try:
        if data and data.get("data") and len(data["data"]) >= 2:
            rows = data["data"][-2:]
            now  = float(rows[-1].get("openInterestUsd") or rows[-1].get("oiUsd"))
            prev = float(rows[-2].get("openInterestUsd") or rows[-2].get("oiUsd"))
            if prev > 0:
                return 100.0 * (now - prev) / prev
    except Exception:
        pass
    return None

def fetch_long_short(sym: str) -> Optional[float]:
    """Return long/short accounts ratio (>1 = more longs)."""
    s = _sym(sym)
    data = _get("/longShortRate", {"symbol": s})
    try:
        if data and data.get("data"):
            return float(data["data"][-1].get("longShortRate") or data["data"][-1].get("ratio"))
    except Exception:
        pass
    return None

def fetch_liq_buckets(sym: str):
    """Return liquidation heatmap buckets (provider-specific)."""
    s = _sym(sym)
    data = _get("/liquidation/heatmap", {"symbol": s})
    return (data or {}).get("data")

def nearest_liq_walls(price: float, buckets) -> Tuple[Optional[float], Optional[float]]:
    """Return distance (%) to closest liq wall below / above price."""
    if not buckets:
        return None, None
    below, above = [], []
    for b in buckets:
        p = float(b.get("price") or b.get("level") or 0)
        v = float(b.get("sizeUsd") or b.get("sum") or 0)
        if p <= 0 or v <= 0:
            continue
        d = abs((p - price) / price) * 100.0
        (below if p < price else above).append(d)
    return (min(below) if below else None, min(above) if above else None)

def deriv_summary(sym: str, price: float) -> Optional[str]:
    """Return a compact contrarian derivatives summary string."""
    if not KEY:
        return None

    fr   = fetch_funding(sym)           # ± per 8h (fraction)
    oi1h = fetch_oi_1h_change(sym)      # % change
    lsr  = fetch_long_short(sym)        # ratio
    liq  = fetch_liq_buckets(sym)
    low, high = nearest_liq_walls(price, liq)

    score = 0
    parts = []

    if fr is not None:
        parts.append(f"Funding {fr*100:+.2f}%/8h")
        if fr > FUND_LO: score += 1
        if fr > FUND_HI: score += 1
        if fr < -FUND_LO: score -= 1
        if fr < -FUND_HI: score -= 1

    if lsr is not None:
        parts.append(f"L/S {lsr:.2f}")
        if lsr > 1.6: score += 1
        if lsr > 2.0: score += 1
        if lsr < 0.6: score -= 1
        if lsr < 0.5: score -= 1

    if oi1h is not None:
        parts.append(f"OI Δ1h {oi1h:+.1f}%")
        if oi1h > OI_1H_T: score += 1
        if oi1h < -OI_1H_T: score -= 1

    if (low is not None) or (high is not None):
        parts.append(f"Liq walls: {('-'+f'{low:.1f}%' if low is not None else '-')}/"
                     f"{('+'+f'{high:.1f}%' if high is not None else '-')}")

    if not parts:
        return None
    crowd = "LONG" if score > 0 else ("SHORT" if score < 0 else "neutral")
    return f"crowd {crowd} {score:+d} | " + " | ".join(parts)

