"""
dashboard/app.py
================
Streamlit app — upload a signal + returns CSV, get instant decay analysis.

Run locally:
    streamlit run dashboard/app.py

Deploy free to Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app → point at dashboard/app.py
    3. Done. Free hosting, auto-redeploys on git push.

TODO (Phase 4):
    - Add multi-signal comparison tab
    - Add crowding proxy upload + CrowdingDetector integration
    - Add download button for the IC curve CSV
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow importing halflife from parent directory when running locally
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
    from halflife import SignalDecayEstimator
    from halflife.utils import forward_returns
except ImportError as e:
    raise ImportError(
        "Install dependencies: pip install -e '.[dev]'"
    ) from e


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="alpha-decay-lab | Signal Half-Life Explorer",
    page_icon="📉",
    layout="wide",
)

st.title("Signal half-life explorer")
st.caption("Upload your signal + returns CSV → get IC decay curve and half-life estimate.")

# ---------------------------------------------------------------------------
# Sidebar: controls
# ---------------------------------------------------------------------------

st.sidebar.header("Parameters")

window = st.sidebar.slider(
    "Rolling IC window (periods)",
    min_value=20, max_value=252, value=63, step=1,
    help="Larger window = smoother IC curve but detects decay more slowly."
)

decay_type = st.sidebar.selectbox(
    "Decay model",
    options=["exponential", "power_law"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**CSV format required:**\n"
    "Two columns: `signal` and `returns`. One row per period. "
    "Returns should already be forward-shifted."
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload signal + returns CSV",
    type=["csv"],
    help="CSV with columns: signal, returns"
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)

        if "signal" not in df.columns or "returns" not in df.columns:
            st.error("CSV must have columns named `signal` and `returns`.")
            st.stop()

        df = df.dropna(subset=["signal", "returns"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows loaded", f"{len(df):,}")
        col2.metric("Signal mean", f"{df['signal'].mean():.4f}")
        col3.metric("Returns mean", f"{df['returns'].mean():.4f}")

        # Run estimator
        with st.spinner("Fitting decay model..."):
            est = SignalDecayEstimator(
                signal=df["signal"],
                returns=df["returns"],
                window=window,
                decay_type=decay_type,
            )
            result = est.fit()

        # Results
        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Half-life", f"{result.half_life:.1f} periods" if not np.isnan(result.half_life) else "n/a")
        m2.metric("IC mean", f"{result.ic_mean:.4f}")
        m3.metric("IC IR", f"{result.ic_ir:.3f}")
        m4.metric("Decay type", result.decay_type)

        # IC curve plot
        st.subheader("IC decay curve")
        fig, ax = plt.subplots(figsize=(10, 3))
        result.plot(ax=ax)
        st.pyplot(fig)
        plt.close()

        # Raw summary
        with st.expander("Full summary"):
            st.code(result.summary())

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

else:
    st.info("Upload a CSV file to get started. Or use the demo below.")

    # Demo with synthetic data
    if st.button("Run demo with synthetic signal"):
        rng = np.random.default_rng(42)
        n = 500
        lam = np.log(2) / 80  # true half-life of 80 periods
        t = np.arange(n)
        ic_t = 0.08 * np.exp(-lam * t)
        signal = rng.standard_normal(n)
        returns = ic_t * signal + np.sqrt(1 - ic_t**2) * rng.standard_normal(n)

        est = SignalDecayEstimator(
            signal=pd.Series(signal),
            returns=pd.Series(returns),
            window=window,
        )
        result = est.fit()

        st.success(f"Demo: estimated half-life = {result.half_life:.1f} periods (true = 80)")

        fig, ax = plt.subplots(figsize=(10, 3))
        result.plot(ax=ax, title="Demo: synthetic decaying signal (true half-life = 80)")
        st.pyplot(fig)
        plt.close()
