# AI options backtest

Backtests 17 option strategies (long puts, put spreads, straddles/strangles, short puts, iron condors,
protective puts, collars, covered calls) on volatile AI-linked stocks
(NVDA, AMD, AVGO, MU, ARM, SMCI, PLTR, CRWV, ORCL, TSLA) plus SMH and QQQ, 2018 onward.
It also stress-tests a set of hedges against today's real option quotes.

```
pip install -r requirements.txt
python run_all.py --refresh     # download prices, calibrate, run full grid + live scenario (~4 min)
python run_all.py --scenario-only   # refresh only the live-chain stress test
python build_report.py          # writes results/report.html
```

## Method (short)
- Historical option prices are **simulated** (Black-Scholes). ATM IV proxy = VIX x (stock RV120 / SPY RV120),
  scaled so mean implied variance = 1.15x realised (1.35x for ETFs); smile fitted to live chains (`aibt/calibrate.py`).
- Costs: 2-5% half-spread per leg + commission. Daily mark-to-market; take-profit rules where noted.
- Every strategy is run from every roll-start offset and at IV x 0.85 / 1.00 / 1.15.
- Long-premium strategies spend a fixed 10%/yr premium budget; short-vol is cash-secured; overlays are fully invested.

## Outputs (`results/`)
- `basket_summary.csv` - equal-weight AI basket, SMH and QQQ metrics per strategy
- `summary_by_ticker.csv` - per ticker / strategy / IV assumption (averaged over start offsets)
- `crash_windows*.csv` - returns inside COVID, 2022, Aug-2024, DeepSeek/tariffs, current sell-off
- `scenario_now.csv` - live-chain stress test (spot shocks -40%..+20% over 14 days)
- `report.html` - the full report

Research only, not financial advice.
