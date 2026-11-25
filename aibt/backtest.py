"""Option-strategy backtest on simulated (Black-Scholes + calibrated IV proxy) option prices.

Each strategy is a list of legs (call/put, strike as % of spot at entry, quantity) plus an optional
stock leg, rolled on a fixed schedule. Trades are marked to market daily so exit rules
(take-profit / stop-loss) and crash-window performance can be measured.

Capital conventions
  debit  : net option buyer.  A fixed premium budget of BUDGET_PA (10%/yr of equity) is spent,
           pro-rata per roll; the rest sits in cash (no interest). Equity = E0 * (1 + f * trade_return).
  credit : net option seller, unlevered: one contract per 100 shares of notional capital (cash-secured).
  stock  : 100% of capital in stock + option overlay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .data import HALF_SPREAD, MIN_HALF_SPREAD, COMMISSION
from .pricing import bs, skew_mult

BUDGET_PA = 0.10


@dataclass(frozen=True)
class Leg:
    cp: int          # +1 call, -1 put
    m: float         # strike / spot at entry
    q: float         # +long / -short (per share of underlying)


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: tuple
    days: int                    # trading days to expiry at entry (= roll period)
    kind: str                    # debit | credit | stock
    stock: float = 0.0
    take_profit: float | None = None   # close when trade return >= this (fraction of capital basis)
    stop_loss: float | None = None     # close when trade return <= -this
    iv_filter: float | None = None     # only enter when IV percentile (1y) <= this
    label: str = ""
    group: str = ""


P, C = -1, +1
STRATEGIES = [
    # --- long volatility / bearish: the "crash" thesis ---------------------------------------
    Strategy("put_atm_1m", (Leg(P, 1.00, 1),), 21, "debit", label="Long ATM put, 1M", group="Bearish / long vol"),
    Strategy("put_90_1m", (Leg(P, 0.90, 1),), 21, "debit", label="Long 10% OTM put, 1M", group="Bearish / long vol"),
    Strategy("put_90_3m", (Leg(P, 0.90, 1),), 63, "debit", label="Long 10% OTM put, 3M", group="Bearish / long vol"),
    Strategy("put_80_3m", (Leg(P, 0.80, 1),), 63, "debit", label="Long 20% OTM put, 3M (tail)", group="Bearish / long vol"),
    Strategy("put_90_3m_tp", (Leg(P, 0.90, 1),), 63, "debit", take_profit=2.0,
             label="10% OTM put 3M, take profit at +200%", group="Bearish / long vol"),
    Strategy("put_90_3m_lowiv", (Leg(P, 0.90, 1),), 63, "debit", iv_filter=0.35, take_profit=2.0,
             label="10% OTM put 3M, only when IV cheap (<35th pct), TP +200%", group="Bearish / long vol"),
    Strategy("pspread_100_85_1m", (Leg(P, 1.00, 1), Leg(P, 0.85, -1)), 21, "debit",
             label="Bear put spread 100/85, 1M", group="Bearish / long vol"),
    Strategy("pspread_95_75_3m", (Leg(P, 0.95, 1), Leg(P, 0.75, -1)), 63, "debit",
             label="Bear put spread 95/75, 3M", group="Bearish / long vol"),
    Strategy("straddle_1m", (Leg(P, 1.00, 1), Leg(C, 1.00, 1)), 21, "debit",
             label="Long ATM straddle, 1M", group="Bearish / long vol"),
    Strategy("strangle_1m", (Leg(P, 0.90, 1), Leg(C, 1.10, 1)), 21, "debit",
             label="Long 10% OTM strangle, 1M", group="Bearish / long vol"),
    Strategy("straddle_1m_lowiv", (Leg(P, 1.00, 1), Leg(C, 1.00, 1)), 21, "debit", iv_filter=0.35,
             label="Long straddle 1M, only when IV cheap", group="Bearish / long vol"),
    # --- short volatility: the other side of the trade ---------------------------------------
    Strategy("short_put_90_1m", (Leg(P, 0.90, -1),), 21, "credit", label="Short 10% OTM put, 1M (cash-secured)",
             group="Short vol (counterpoint)"),
    Strategy("iron_condor_1m", (Leg(P, 0.80, 1), Leg(P, 0.90, -1), Leg(C, 1.10, -1), Leg(C, 1.20, 1)), 21, "credit",
             take_profit=0.5, label="Iron condor 80/90/110/120, 1M", group="Short vol (counterpoint)"),
    # --- stock + overlays --------------------------------------------------------------------
    Strategy("stock", (), 21, "stock", stock=1, label="Buy & hold stock", group="Stock + overlay"),
    Strategy("protective_put", (Leg(P, 0.90, 1),), 63, "stock", stock=1,
             label="Stock + 10% OTM put, 3M rolls", group="Stock + overlay"),
    Strategy("collar", (Leg(P, 0.90, 1), Leg(C, 1.15, -1)), 63, "stock", stock=1,
             label="Collar: stock + 90% put − 115% call, 3M", group="Stock + overlay"),
    Strategy("covered_call", (Leg(C, 1.10, -1),), 21, "stock", stock=1,
             label="Covered call 110%, 1M", group="Stock + overlay"),
]
BY_NAME = {s.name: s for s in STRATEGIES}


@dataclass
class Market:
    ticker: str
    S: np.ndarray
    iv: np.ndarray        # ATM IV proxy (calibrated)
    ivpct: np.ndarray     # trailing 1y percentile of iv
    r: np.ndarray
    dates: pd.DatetimeIndex
    a: float
    b: float
    hs: float


def build_market(d: dict, ticker: str, cal: dict, iv_mult: float = 1.0, start: str = "2018-01-01") -> Market:
    from .data import iv_proxy_unscaled
    px = d["close"][ticker].dropna()
    iv = iv_proxy_unscaled(d, ticker) * cal[ticker]["k"] * iv_mult
    df = pd.DataFrame({"S": px, "iv": iv, "r": d["rf"].reindex(px.index).ffill().fillna(0.0)}).dropna()
    df["ivpct"] = df["iv"].rolling(252, min_periods=120).rank(pct=True)
    df = df[df.index >= start]
    return Market(ticker, df["S"].values, df["iv"].values, df["ivpct"].fillna(0.5).values, df["r"].values,
                  df.index, cal[ticker]["skew_a"], cal[ticker]["skew_b"], HALF_SPREAD[ticker])


def _cost(px: np.ndarray, S: np.ndarray, hs: float) -> np.ndarray:
    return np.maximum(MIN_HALF_SPREAD * S, hs * px) + COMMISSION * S


def run(mkt: Market, st: Strategy, offset: int = 0):
    """Returns (trades DataFrame, daily equity Series)."""
    n = len(mkt.S)
    H = st.days
    eq = np.full(n, np.nan)
    E = 1.0
    trades = []
    i0 = offset
    eq[:i0 + 1] = 1.0
    while i0 + 1 < n:
        hold = min(H, n - 1 - i0)
        if st.iv_filter is not None and mkt.ivpct[i0] > st.iv_filter:
            eq[i0:i0 + hold + 1] = E  # sit in cash until next scheduled check
            i0 += hold
            continue
        S0 = mkt.S[i0]
        js = np.arange(i0, i0 + hold + 1)
        S = mkt.S[js]
        Tleft = (H - (js - i0)) / 252.0
        legs_mid = []
        for lg in st.legs:
            K = lg.m * S0
            sig = mkt.iv[js] * skew_mult(K, S, np.maximum(Tleft, 1 / 252), mkt.a, mkt.b)
            legs_mid.append(bs(S, K, Tleft, mkt.r[js], sig, lg.cp))
        q = np.array([lg.q for lg in st.legs])
        mids = np.array(legs_mid) if st.legs else np.zeros((0, len(js)))
        costs = _cost(mids, S[None, :], mkt.hs) if st.legs else mids
        # value of the position if liquidated on day j (pay spread on every leg)
        liq = (q[:, None] * mids - np.abs(q)[:, None] * costs).sum(0) + st.stock * S
        if st.legs:
            # at expiry: cash-settle at intrinsic, only commission on ITM legs
            itm = mids[:, -1] > 0
            if hold == H:
                liq[-1] = (q * mids[:, -1]).sum() - (np.abs(q) * COMMISSION * S[-1] * itm).sum() + st.stock * S[-1]
        entry = (q * mids[:, 0] + np.abs(q) * costs[:, 0]).sum() + st.stock * S0 if st.legs else st.stock * S0
        if st.kind == "debit":
            basis = entry
            f = BUDGET_PA * H / 252.0
        elif st.kind == "credit":
            basis = S0                      # cash-secured notional
            f = 1.0
        else:
            basis = entry
            f = 1.0
        ret_path = (liq - entry) / basis
        exit_k = len(js) - 1
        reason = "expiry" if hold == H else "end_of_data"
        for k in range(1, len(js) - 1):
            rr = ret_path[k]
            if st.take_profit is not None:
                tp = st.take_profit if st.kind == "debit" else st.take_profit * (-entry) / basis
                if rr >= tp:
                    exit_k, reason = k, "take_profit"
                    break
            if st.stop_loss is not None and rr <= -st.stop_loss:
                exit_k, reason = k, "stop"
                break
        ret_path = ret_path.copy()
        ret_path[exit_k:] = ret_path[exit_k]
        eq[js] = E * (1.0 + f * ret_path)
        trades.append(dict(entry=mkt.dates[i0], exit=mkt.dates[js[exit_k]], S0=S0, S_exit=S[exit_k],
                           iv0=mkt.iv[i0], cost=entry, ret=ret_path[exit_k], reason=reason,
                           port_ret=f * ret_path[exit_k]))
        E = eq[js[-1]]
        i0 += hold
    return pd.DataFrame(trades), pd.Series(eq, index=mkt.dates).ffill()


def metrics(eq: pd.Series, trades: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    eq = eq.dropna()
    rets = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = rets.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    out = dict(cagr=cagr, vol=vol, sharpe=(rets.mean() * 252) / vol if vol > 0 else np.nan, max_dd=dd,
               total=eq.iloc[-1] - 1, n_trades=len(trades),
               win_rate=(trades["ret"] > 0).mean() if len(trades) else np.nan,
               avg_trade=trades["ret"].mean() if len(trades) else np.nan,
               med_trade=trades["ret"].median() if len(trades) else np.nan,
               best_trade=trades["ret"].max() if len(trades) else np.nan,
               worst_trade=trades["ret"].min() if len(trades) else np.nan)
    if bench is not None:
        b = bench.pct_change().reindex(rets.index)
        out["corr_bench"] = rets.corr(b)
    return out
