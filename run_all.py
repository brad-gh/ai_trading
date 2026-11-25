"""Full run: historical grid backtest + crash-window analysis + live-chain stress test.

    python run_all.py            # uses cached prices
    python run_all.py --refresh  # re-download prices and recalibrate
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
import pandas as pd

from aibt.data import load, AI_TICKERS, UNIVERSE
from aibt.calibrate import calibrate
from aibt.backtest import STRATEGIES, build_market, run, metrics
from aibt import scenario

IV_MULTS = [0.85, 1.00, 1.15]
OUT = "results"


def crash_windows(d):
    smh = d["close"]["SMH"]
    recent = smh[-90:]
    w = {
        "COVID crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
        "2022 tech bear market": ("2021-11-19", "2022-10-13"),
        "AI unwind (Jul–Aug 2024)": ("2024-07-10", "2024-08-07"),
        "DeepSeek + tariffs (Jan–Apr 2025)": ("2025-01-23", "2025-04-08"),
        f"Current sell-off (from {recent.idxmax():%b %d %Y})": (f"{recent.idxmax():%Y-%m-%d}", f"{smh.index[-1]:%Y-%m-%d}"),
    }
    return w


def window_ret(eq: pd.Series, a, b):
    s = eq.loc[:b]
    s0 = eq.loc[:a]
    if len(s0) == 0 or len(s) == 0 or eq.index[0] > pd.Timestamp(a):
        return np.nan
    return s.iloc[-1] / s0.iloc[-1] - 1


def main(refresh=False, scenario_only=False):
    if scenario_only:
        scenario.run(UNIVERSE).to_csv(f"{OUT}/scenario_now.csv", index=False)
        return
    t0 = time.time()
    d = load(refresh=refresh)
    if refresh:
        calibrate()
    cal = json.load(open(f"{OUT}/calibration.json"))
    bench = d["close"]["QQQ"]
    windows = crash_windows(d)

    rows, wrows, curves = [], [], {}
    for tk in UNIVERSE:
        for ivm in IV_MULTS:
            mkt = build_market(d, tk, cal, ivm)
            for st in STRATEGIES:
                step = 1 if st.days <= 21 else 3
                eqs = []
                for off in range(0, st.days, step):
                    tr, eq = run(mkt, st, off)
                    m = metrics(eq, tr, bench)
                    m.update(ticker=tk, strategy=st.name, iv_mult=ivm, offset=off)
                    rows.append(m)
                    eqs.append(eq)
                    if ivm == 1.0:
                        for wn, (a, b) in windows.items():
                            wrows.append(dict(ticker=tk, strategy=st.name, window=wn, offset=off,
                                              ret=window_ret(eq, a, b)))
                if ivm == 1.0:
                    curves[(tk, st.name)] = pd.concat(eqs, axis=1).mean(axis=1)
        print(f"{tk} done ({time.time()-t0:.0f}s)")

    runs = pd.DataFrame(rows)
    runs.to_csv(f"{OUT}/all_runs.csv.gz", index=False)
    agg = (runs.groupby(["ticker", "strategy", "iv_mult"])
           .agg(cagr=("cagr", "mean"), cagr_p10=("cagr", lambda x: x.quantile(0.1)),
                cagr_p90=("cagr", lambda x: x.quantile(0.9)), vol=("vol", "mean"), sharpe=("sharpe", "mean"),
                max_dd=("max_dd", "mean"), win_rate=("win_rate", "mean"), avg_trade=("avg_trade", "mean"),
                med_trade=("med_trade", "mean"), best_trade=("best_trade", "mean"),
                worst_trade=("worst_trade", "mean"), corr_qqq=("corr_bench", "mean"),
                n_trades=("n_trades", "mean"), n_runs=("cagr", "size"))
           .reset_index())
    agg.to_csv(f"{OUT}/summary_by_ticker.csv", index=False)

    # Equal-weight AI basket (10 single names), base IV, start-date-averaged curves.
    bask = {}
    for st in STRATEGIES:
        rets = pd.concat([curves[(tk, st.name)].pct_change() for tk in AI_TICKERS], axis=1)
        bask[st.name] = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    bask = pd.DataFrame(bask)
    for tk in ["SMH", "QQQ"]:
        for st in STRATEGIES:
            bask[f"{tk}:{st.name}"] = curves[(tk, st.name)]
    bask.to_csv(f"{OUT}/equity_curves.csv")
    bsum = []
    for c in bask.columns:
        e = bask[c].dropna()
        m = metrics(e, pd.DataFrame({"ret": []}), bench)
        m["series"] = c
        bsum.append(m)
    pd.DataFrame(bsum)[["series", "cagr", "vol", "sharpe", "max_dd", "corr_bench"]].to_csv(
        f"{OUT}/basket_summary.csv", index=False)

    w = pd.DataFrame(wrows)
    w.groupby(["window", "ticker", "strategy"])["ret"].mean().reset_index().to_csv(
        f"{OUT}/crash_windows.csv", index=False)
    wb = []
    for wn, (a, b) in windows.items():
        for c in bask.columns:
            wb.append(dict(window=wn, series=c, ret=window_ret(bask[c].dropna(), a, b)))
    pd.DataFrame(wb).to_csv(f"{OUT}/crash_windows_basket.csv", index=False)
    json.dump({k: list(v) for k, v in windows.items()}, open(f"{OUT}/windows.json", "w"), indent=1)

    print("historical grid done", f"{time.time()-t0:.0f}s, {len(runs)} backtests")
    sc = scenario.run(UNIVERSE)
    sc.to_csv(f"{OUT}/scenario_now.csv", index=False)
    print("all done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv, scenario_only="--scenario-only" in sys.argv)
