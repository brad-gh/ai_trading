"""Calibrate the IV proxy level (k) and smile (a, b) per ticker against today's live chains."""
from __future__ import annotations

import datetime as dt
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

from .data import UNIVERSE, load, iv_proxy_unscaled
from .pricing import implied_vol

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "calibration.json")


def _pick_expiry(expiries, today, target_days):
    ds = [(abs((pd.Timestamp(e) - today).days - target_days), e) for e in expiries]
    return min(ds)[1]


def chain_ivs(ticker: str, target_days: int = 30, r: float = 0.04):
    t = yf.Ticker(ticker)
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    today = pd.Timestamp(dt.date.today())
    exp = _pick_expiry(t.options, today, target_days)
    T = max((pd.Timestamp(exp) - today).days, 1) / 365.0
    ch = t.option_chain(exp)
    rows = []
    live_quotes = (ch.puts["bid"] > 0).sum() + (ch.calls["bid"] > 0).sum() > 10
    for cp, df in ((1, ch.calls), (-1, ch.puts)):
        for _, o in df.iterrows():
            if live_quotes:
                if not (o["bid"] > 0 and o["ask"] >= o["bid"]):
                    continue
                mid = 0.5 * (o["bid"] + o["ask"])
                bid, ask, src = o["bid"], o["ask"], "bid/ask"
            else:
                # Outside market hours Yahoo blanks bid/ask: fall back to last trade + synthetic spread.
                last = o.get("lastPrice", 0.0)
                ltd = pd.Timestamp(o.get("lastTradeDate")).tz_localize(None) if pd.notna(o.get("lastTradeDate")) else None
                if not (last > 0.05 and ltd is not None and (today - ltd.normalize()).days <= 4):
                    continue
                mid = float(last)
                hs = max(0.01, 0.04 * mid)
                bid, ask, src = mid - hs, mid + hs, "last"
            K = float(o["strike"])
            otm = (cp == -1 and K <= spot) or (cp == 1 and K >= spot)
            iv = implied_vol(mid, spot, K, T, r, cp)
            rows.append(dict(cp=cp, K=K, mid=mid, bid=bid, ask=ask, iv=iv, otm=otm, src=src,
                             oi=o.get("openInterest", np.nan)))
    df = pd.DataFrame(rows, columns=["cp", "K", "mid", "bid", "ask", "iv", "otm", "src", "oi"]).dropna(subset=["iv"])
    return spot, exp, T, df


TARGET_VAR_RATIO = {"SMH": 1.35, "QQQ": 1.35}  # implied / realised variance; default below
DEFAULT_VAR_RATIO = 1.15


def anchored_k(d: dict, tk: str) -> float:
    """Scale so that historical mean model implied variance = target x mean forward 21d realised variance."""
    px = d["close"][tk].dropna()
    iv = iv_proxy_unscaled(d, tk)
    lr = np.log(px).diff()
    fwd_var = (lr[::-1].rolling(21).var()[::-1].shift(-1)) * 252
    x = pd.concat([iv, fwd_var], axis=1).dropna()
    target = TARGET_VAR_RATIO.get(tk, DEFAULT_VAR_RATIO)
    return float(np.sqrt(target * x.iloc[:, 1].mean() / (x.iloc[:, 0] ** 2).mean()))


def calibrate(r: float = 0.04) -> dict:
    d = load()
    res = {}
    for tk in UNIVERSE:
        try:
            spot, exp, T, df = chain_ivs(tk, 30, r)
        except Exception as e:  # noqa: BLE001
            print(f"{tk}: chain failed ({e})")
            continue
        near = df.iloc[(df["K"] - spot).abs().argsort()[:4]]
        iv_atm = float(near["iv"].median())
        otm = df[df["otm"]].copy()
        otm["m"] = np.log(otm["K"] / spot) / np.sqrt(T)
        otm = otm[(otm["m"].abs() < 2.5) & (otm["mid"] > 0.05)]
        y = otm["iv"] / iv_atm
        a, b = 0.0, 0.05
        if len(otm) >= 8:
            coef = np.polyfit(otm["m"], y, 2)  # c2 m^2 + c1 m + c0
            b, a = float(np.clip(coef[0], 0.0, 0.3)), float(np.clip(coef[1], -0.4, 0.3))
        proxy_today = float(iv_proxy_unscaled(d, tk).dropna().iloc[-1])
        k_live = float(np.clip(iv_atm / proxy_today, 0.3, 3.0))
        k = anchored_k(d, tk)
        res[tk] = dict(spot=spot, expiry=exp, T=T, iv_atm_live=iv_atm, proxy_unscaled=proxy_today,
                       k=k, k_live=k_live, model_iv_today=k * proxy_today,
                       skew_a=a, skew_b=b, n_quotes=int(len(otm)), skew_fallback=len(otm) < 8)
        print(f"{tk:5s} spot {spot:8.2f} liveATM {iv_atm:.1%} modelIV today {k*proxy_today:.1%} "
              f"k={k:.2f} (live k={k_live:.2f}) a={a:+.3f} b={b:.3f} n={len(otm)}")
    singles = [v for t, v in res.items() if not v["skew_fallback"] and t not in TARGET_VAR_RATIO]
    pa, pb = float(np.median([v["skew_a"] for v in singles])), float(np.median([v["skew_b"] for v in singles]))
    for v in res.values():
        if v["skew_fallback"]:
            v["skew_a"], v["skew_b"] = pa, pb
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    calibrate()
