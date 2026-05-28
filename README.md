# alpha-decay-lab

> A framework for measuring, modelling, and extending the half-life of microstructure trading signals.

When a new alpha signal is discovered order capital flows toward it. Arbitrageurs trade against it. Market makers adapt. Within weeks the edge can collapse entirely. This repo gives you the tools to measure that decay precisely, understand why it happens, and build adaptive signal weighting that keeps you ahead of the crowding curve.

---

## What's in here

| Module | What it does |
|--------|-------------|
| `halflife/` | Installable Python package — IC decay curves, half-life estimation, crowding diagnostics, adaptive weighting |
| `notebooks/` | End-to-end research notebooks from data ingestion to paper-ready results |
| `dashboard/` | Streamlit app — upload your signal + returns, get instant decay analysis |
| `paper/` | LaTeX source for the accompanying SSRN empirical note |
| `tests/` | pytest suite for all estimators |

---

## Quickstart

```bash
git clone https://github.com/DekosZero/alpha-decay-lab.git
cd alpha-decay-lab
pip install -e ".[dev]"
```

Run the decay estimator on a toy signal:

```python
from halflife import SignalDecayEstimator

estimator = SignalDecayEstimator(signal=my_signal_series, returns=my_returns_series)
result = estimator.fit()

print(result.half_life)       # float, in periods
print(result.decay_type)      # 'exponential' or 'power_law'
print(result.ic_curve)        # pd.Series of rolling IC
result.plot()                 # IC decay chart with half-life annotation
```

---

## Data sources

This project is designed to work with **free, publicly accessible data** first:

- **Binance perpetual swaps** — tick-level order book snapshots via REST API, free, no account required for public endpoints. Start here.
- **LOBSTER** — limit order book data for NASDAQ equities. Free for academic use at [lobsterdata.com](https://lobsterdata.com).
- **WRDS TAQ** — NYSE Trade and Quote data. Requires university library access.

See `notebooks/01_data_pipeline.ipynb` for ingestion code for all three.

---

## Project phases

### Phase 1 — Literature review & data pipeline (weeks 1–4)
Read the five core papers listed in `paper/references.bib`. Stand up the Binance data pipeline in `notebooks/01`. Understand what IC, IR, and VWAP-based returns actually measure before writing any signal code.

### Phase 2 — Core estimators (weeks 5–10)
Build `halflife/estimators.py`. The minimum viable deliverable is `SignalDecayEstimator.fit()` that takes a signal + returns and outputs a half-life with confidence interval. Test against synthetic signals with known decay properties.

### Phase 3 — Benchmark study (weeks 11–14)
Run the estimators across canonical microstructure signals (OFI, trade sign, short-term reversal) on Binance BTC/ETH perpetuals and LOBSTER AAPL/MSFT. Produce the results tables in `notebooks/06_benchmark_results.ipynb`.

### Phase 4 — Dashboard & docs (weeks 15–17)
Build the Streamlit app in `dashboard/app.py`. Write docstrings. Publish to Streamlit Cloud (free tier).

### Phase 5 — SSRN preprint (weeks 18–19)
Write up the benchmark results. A 10-page empirical note is sufficient. Submit to SSRN under Quantitative Finance.

---

## Core references

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Chapters 4–5 on feature importance and cross-validation.
- Bouchaud, J-P. et al. (2017). *Trades, Quotes and Prices*. Cambridge. — Chapter 11 on price impact and information.
- Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
- Glosten, L. & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71–100.
- Moody, J. & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875–889.

---

## License

MIT
