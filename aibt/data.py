"""Price / rates / VIX download with a local pickle cache."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf

# Volatile AI-linked names + two index benchmarks.
AI_TICKERS = ["NVDA", "AMD", "AVGO", "MU", "ARM", "SMCI", "PLTR", "CRWV", "ORCL", "TSLA"]
INDEX_TICKERS = ["SMH", "QQQ"]
UNIVERSE = AI_TICKERS + INDEX_TICKERS
AUX = ["SPY", "^VIX", "^IRX"]

# Rough option-liquidity tiers -> half bid/ask spread as fraction of option mid.
HALF_SPREAD = {t: 0.02 for t in UNIVERSE}
HALF_SPREAD.update({"SMCI": 0.04, "CRWV": 0.05, "ARM": 0.04, "ORCL": 0.03, "MU": 0.03})
# Prices are split-adjusted, so fixed-dollar frictions are expressed as a fraction of spot instead.
MIN_HALF_SPREAD = 0.0002  # ~ $0.025 tick-floor on a $125 stock
COMMISSION = 0.00005      # ~ $0.65 / contract on a $130 stock, per leg per side

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "prices.pkl")


def load(start: str = "2017-01-01", refresh: bool = False) -> dict[str, pd.DataFrame | pd.Series]:
    if os.path.exists(CACHE) and not refresh:
        return pd.read_pickle(CACHE)
    raw = yf.download(UNIVERSE + AUX, start=start, progress=False, auto_adjust=True)
    close = raw["Close"].dropna(how="all")
    out = {
        "close": close[UNIVERSE],
        "spy": close["SPY"],
        "vix": close["^VIX"].ffill(),
        "rf": (close["^IRX"].ffill() / 100.0).clip(lower=0.0),
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def realized_vol(px: pd.Series, window: int) -> pd.Series:
    lr = np.log(px).diff()
    return lr.rolling(window, min_periods=max(20, window // 3)).std() * np.sqrt(252)


def iv_proxy_unscaled(d: dict, ticker: str) -> pd.Series:
    """ATM 30d implied-vol proxy before calibration:
        VIX_t * (RV120_stock / RV120_SPY)
    VIX supplies the forward-looking market regime (it jumps on the day of a shock),
    the RV ratio supplies the stock's vol level relative to the market."""
    px = d["close"][ticker].dropna()
    ratio = (realized_vol(px, 120) / realized_vol(d["spy"], 120).reindex(px.index)).clip(0.7, 8.0)
    return (d["vix"].reindex(px.index) / 100.0 * ratio).rename(ticker)
