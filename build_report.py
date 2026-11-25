"""Build results/report.html from the CSVs written by run_all.py."""
from __future__ import annotations

import json
import datetime as dt
import numpy as np
import pandas as pd

from aibt.backtest import STRATEGIES, BY_NAME
from aibt.data import AI_TICKERS

R = "results"


def _clean(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def main():
    cal = json.load(open(f"{R}/calibration.json"))
    windows = json.load(open(f"{R}/windows.json"))
    curves = pd.read_csv(f"{R}/equity_curves.csv", index_col=0, parse_dates=True)
    bsum = pd.read_csv(f"{R}/basket_summary.csv").set_index("series")
    wb = pd.read_csv(f"{R}/crash_windows_basket.csv")
    byt = pd.read_csv(f"{R}/summary_by_ticker.csv")
    sc = pd.read_csv(f"{R}/scenario_now.csv")

    wk = curves.resample("W-FRI").last()
    long_vol = ["put_90_3m", "put_80_3m", "put_90_3m_tp", "straddle_1m"]
    overlay = ["stock", "protective_put", "collar", "short_put_90_1m"]

    def series(names, prefix=""):
        out = []
        for n in names:
            s = wk[prefix + n].dropna()
            out.append(dict(key=n, label=BY_NAME[n].label,
                            pts=[[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.items()]))
        return out

    sens = byt.pivot_table(index="strategy", columns="iv_mult", values="cagr", aggfunc="median")
    rows = []
    for st in STRATEGIES:
        r = dict(key=st.name, label=st.label, group=st.group, kind=st.kind)
        for pre, tag in (("", "ai"), ("SMH:", "smh"), ("QQQ:", "qqq")):
            m = bsum.loc[pre + st.name]
            r[tag] = {k: _clean(float(m[k])) for k in ("cagr", "max_dd", "sharpe", "corr_bench")}
            r[tag]["win"] = {w: _clean(float(wb[(wb.series == pre + st.name) & (wb.window == w)]["ret"].iloc[0]))
                             for w in windows}
        r["sens"] = {str(k): _clean(float(sens.loc[st.name, k])) for k in sens.columns}
        rows.append(r)

    tick = byt[byt.iv_mult == 1.0].pivot(index="strategy", columns="ticker", values="cagr")
    tick_rows = [dict(key=st.name, label=st.label,
                      vals={t: _clean(float(tick.loc[st.name, t])) for t in tick.columns}) for st in STRATEGIES]

    scen = []
    for _, x in sc.iterrows():
        scen.append(dict(ticker=x.ticker, dte=int(x.dte), expiry=x.expiry, structure=x.structure,
                         strikes=str(x.strikes), cost=float(x.cost), cost_pct=float(x.cost_pct_spot),
                         spot=float(x.spot), iv=str(x.iv_legs), src=str(x.get("quote_source", "")),
                         rets={c.replace("ret_", ""): max(-1.0, float(x[c])) for c in sc.columns if c.startswith("ret_")}))
    src = sc["quote_source"].mode().iloc[0] if "quote_source" in sc else "bid/ask"

    ivtab = [dict(t=t, live=_clean(v.get("iv_atm_live")), model=_clean(v.get("model_iv_today"))) for t, v in cal.items()]
    data = dict(
        generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        start=str(curves.index[0].date()), end=str(curves.index[-1].date()),
        windows=windows, longvol=series(long_vol), overlay=series(overlay),
        rows=rows, tick_rows=tick_rows, tickers=list(tick.columns), scen=scen, quote_source=src,
        ivtab=ivtab, n_backtests=int(len(pd.read_csv(f"{R}/all_runs.csv.gz", usecols=["cagr"]))),
        ai_tickers=AI_TICKERS,
    )
    tpl = open("report_template.html").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open(f"{R}/report.html", "w").write(html)
    print("wrote results/report.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
