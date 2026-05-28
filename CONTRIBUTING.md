# Contributing to alpha-decay-lab

Contributions welcome. This is a research project so the bar for a useful PR is "does it help someone measure or understand signal decay better."

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/alpha-decay-lab.git
cd alpha-decay-lab
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v --cov=halflife
```

## Code style

```bash
black halflife/ tests/
ruff check halflife/ tests/
```

## What makes a good contribution

- New signal features in `halflife/features.py` — especially anything on LOB depth, cancellation rates, or toxicity measures
- Better confidence interval methods for the IC decay estimator
- Additional data connectors in `notebooks/01_data_pipeline.ipynb`
- Fixes to the Streamlit dashboard
- Any empirical result that challenges or extends the benchmark findings

## What to avoid

- Adding heavy dependencies (torch, tensorflow) to the core `halflife` package — keep it lightweight
- Committing raw data files to the repo
- Hard-coded API keys or credentials anywhere

## Opening an issue

If you find a bug in an estimator, include: the signal you were using, the length of the series, and the error message or unexpected output.
