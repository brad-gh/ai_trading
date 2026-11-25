"""Forward-looking stress test on TODAY's real option quotes.

Price each structure from the live chain (buy at ask, sell at bid), then reprice after a spot shock
with Black-Scholes using each strike's own quoted IV (sticky-strike), shifted by a vol response.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import yfinance as yf

from .calibrate import chain_ivs
from .pricing import bs

SHOCKS = [-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20]
HORIZON_DAYS = 14  # calendar days until we mark the position


def vol_response(shock: float) -> float:
    """Multiplicative IV change. Down-moves raise IV (+12% of IV per 10% drop), up-moves and
    'nothing happens' bleed it (current IVs are well above long-run averages)."""
    return float(np.clip(1.0 - 1.2 * shock, 0.75, 1.8)) if shock < 0 else float(max(0.75, 0.9 - 0.5 * shock))


STRUCTS = {
    "Long ATM put": [(-1, 1.00, 1)],
    "Long 10% OTM put": [(-1, 0.90, 1)],
    "Long 20% OTM put": [(-1, 0.80, 1)],
    "Bear put spread 100/80": [(-1, 1.00, 1), (-1, 0.80, -1)],
    "Long ATM straddle": [(-1, 1.00, 1), (1, 1.00, 1)],
}


def _nearest(df, cp, K):
    sub = df[(df["cp"] == cp) & (df["ask"] > 0)]
    return sub.iloc[(sub["K"] - K).abs().argsort().iloc[0]]


def run(tickers, target_days=(35, 95), r=0.04):
    rows = []
    for tk in tickers:
        for td in target_days:
            df = None
            for attempt in range(4):
                try:
                    spot, exp, T, df = chain_ivs(tk, td, r)
                    if len(df) and (df["cp"] == -1).sum() >= 5:
                        break
                except Exception as e:  # noqa: BLE001
                    print(tk, td, "attempt", attempt, "failed", e)
                df = None
                time.sleep(3 * (attempt + 1))
            if df is None:
                continue
            T_after = max(T - HORIZON_DAYS / 365.0, 1 / 365)
            for name, legs in STRUCTS.items():
                q = [_nearest(df, cp, m * spot) for cp, m, _ in legs]
                # buy at ask, sell at bid
                cost = sum(o["ask"] * n if n > 0 else o["bid"] * n for o, (_, _, n) in zip(q, legs))
                if cost <= 0:
                    continue
                row = dict(ticker=tk, quote_source=q[0]["src"], expiry=exp, dte=int(round(T * 365)), structure=name, spot=spot,
                           strikes="/".join(f"{o['K']:g}" for o in q), cost=cost, cost_pct_spot=cost / spot,
                           iv_legs="/".join(f"{o['iv']:.0%}" for o in q))
                for sh in SHOCKS:
                    S1 = spot * (1 + sh)
                    vm = vol_response(sh)
                    val = 0.0
                    for o, (cp, _, n) in zip(q, legs):
                        mid = float(bs(S1, o["K"], T_after, r, o["iv"] * vm, cp)[0])
                        hs = max(0.01, 0.5 * (o["ask"] - o["bid"]))   # pay half-spread to exit
                        val += n * mid - abs(n) * hs
                    row[f"ret_{int(sh*100):+d}%"] = val / cost - 1
                rows.append(row)
            print(f"{tk} {exp} done")
    return pd.DataFrame(rows)
