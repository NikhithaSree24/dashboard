# competitor_dashboard.py
"""
Competitor Intelligence Dashboard
Single-file Streamlit app implementing Task 4 requirements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

st.set_page_config(page_title="Competitor Intelligence Dashboard", layout="wide")

# ----------------------------
# Utility & Demo Data
# ----------------------------
@st.cache_data
def generate_demo_data(days=180, competitors=None):
    """Return a demo DataFrame with columns: date, competitor, sector, sentiment_score, mentions, share_of_voice"""
    if competitors is None:
        competitors = [
            ("AlphaLabs", "Fintech"),
            ("BetaAI", "AI"),
            ("GammaSys", "AI"),
            ("DeltaCorp", "Fintech"),
            ("EpsilonX", "AI"),
        ]
    records = []
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    rng = np.random.default_rng(42)
    for comp, sector in competitors:
        base_sent = rng.normal(loc=0.1 if "AI" in sector else 0.0, scale=0.05)
        base_mentions = rng.integers(50, 200)
        sov_base = rng.uniform(0.05, 0.35)
        for i, d in enumerate(dates):
            # Smooth trend + occasional shock
            trend = 0.0006 * i  # small upward trend
            seasonal = 0.05 * np.sin(i * 2 * np.pi / 30)
            shock = rng.choice([0, 0, 0, -0.2, 0.2], p=[0.7,0.1,0.1,0.05,0.05])
            sentiment = base_sent + trend + seasonal + rng.normal(0, 0.03) + shock * (rng.random() < 0.02)
            mentions = int(base_mentions + 5 * i/30 + rng.normal(0, 15))
            share_of_voice = max(0.01, sov_base + 0.0005 * i + 0.02 * np.sin(i * 2*np.pi/45) + rng.normal(0, 0.02))
            records.append({
                "date": d.date(),
                "competitor": comp,
                "sector": sector,
                "sentiment_score": float(np.clip(sentiment, -1, 1)),
                "mentions": max(0, mentions),
                "share_of_voice": float(share_of_voice)
            })
    df = pd.DataFrame.from_records(records)
    # normalize share_of_voice per day so sum is ~1
    df_sum = df.groupby("date")["share_of_voice"].transform("sum")
    df["share_of_voice"] = df["share_of_voice"] / df_sum
    return df

# ----------------------------
# Data ingestion & validation
# ----------------------------
st.title("🔎 Strategic Intelligence — Competitor Dashboard")
st.markdown("Upload a CSV with columns: `date, competitor, sector, sentiment_score, mentions, share_of_voice`"
            " — or use demo data to explore features.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo data", value=True if uploaded is None else False)

if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=["date"])
        df["date"] = df["date"].dt.date
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif use_demo:
    df = generate_demo_data(days=240)
else:
    st.info("Upload a CSV or enable Demo Data from the sidebar.")
    st.stop()

# Basic validation
required_cols = {"date", "competitor", "sector", "sentiment_score", "mentions", "share_of_voice"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV missing required columns. Require: {required_cols}")
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters & Settings")
sectors = sorted(df["sector"].unique())
selected_sector = st.sidebar.multiselect("Sector (filter)", options=sectors, default=sectors)
competitors = sorted(df[df["sector"].isin(selected_sector)]["competitor"].unique())
selected_competitors = st.sidebar.multiselect("Competitors", options=competitors, default=competitors[:3])
date_min = df["date"].min()
date_max = df["date"].max()
start_date, end_date = st.sidebar.date_input("Date range", [date_min, date_max])
smooth_window = st.sidebar.slider("Smoothing window (days) for trend lines", 1, 21, 7)
alert_drop_pct = st.sidebar.slider("Alert: rolling drop threshold (%)", 1, 50, 20)
anomaly_contamination = st.sidebar.slider("Anomaly sensitivity (contamination)", 0.001, 0.2, 0.02)

filtered = df[
    df["sector"].isin(selected_sector) &
    df["competitor"].isin(selected_competitors) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
].copy()

if filtered.empty:
    st.warning("No data for selected filters. Adjust filters or date range.")
    st.stop()

# Derived aggregated timeseries per competitor
def pivot_metric(metric):
    pivot = filtered.pivot_table(index="date", columns="competitor", values=metric, aggfunc="mean").fillna(method="ffill").fillna(0)
    return pivot

sent_pivot = pivot_metric("sentiment_score")
mentions_pivot = pivot_metric("mentions")
sov_pivot = pivot_metric("share_of_voice")

# ----------------------------
# Top row: Leaderboard & Quick Stats
# ----------------------------
col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    st.subheader("🏆 Leaderboard (by avg sentiment)")
    avg_sent = filtered.groupby("competitor")["sentiment_score"].mean().sort_values(ascending=False)
    top_tbl = pd.DataFrame({
        "Avg Sentiment": avg_sent.round(3),
        "Avg Mentions": filtered.groupby("competitor")["mentions"].mean().round(0),
        "Avg ShareOfVoice": filtered.groupby("competitor")["share_of_voice"].mean().round(3)
    }).loc[selected_competitors]
    st.table(top_tbl)

with col2:
    st.subheader("📈 Quick KPI")
    col2_k1, col2_k2 = st.columns(2)
    with col2_k1:
        avg_mentions_overall = int(filtered["mentions"].mean())
        st.metric("Avg Mentions (per day)", f"{avg_mentions_overall}")
    with col2_k2:
        avg_sov = float(filtered["share_of_voice"].mean())
        st.metric("Avg Share of Voice", f"{avg_sov:.3f}")

with col3:
    st.subheader("📊 Sector Snapshot")
    sector_stats = filtered.groupby("sector").agg({
        "sentiment_score": "mean",
        "mentions": "sum"
    }).round(3)
    st.table(sector_stats)

# ----------------------------
# Main visualizations (Trajectories & Area)
# ----------------------------
st.markdown("---")
st.subheader("Competitor Trajectories & Market View")

# Trajectories: sentiment
st.markdown("**Sentiment Trajectories (smoothed)**")
fig, ax = plt.subplots(figsize=(12, 4))
for comp in sent_pivot.columns:
    series = sent_pivot[comp].rolling(smooth_window, min_periods=1).mean()
    ax.plot(series.index, series.values, marker='o', linewidth=1, label=comp)
ax.set_xlabel("Date")
ax.set_ylabel("Sentiment Score")
ax.set_title("Smoothed Sentiment Trajectories")
ax.legend()
plt.xticks(rotation=40)
st.pyplot(fig)

# Trajectories: mentions
st.markdown("**Mentions Over Time**")
fig2, ax2 = plt.subplots(figsize=(12, 4))
for comp in mentions_pivot.columns:
    series = mentions_pivot[comp].rolling(smooth_window, min_periods=1).mean()
    ax2.plot(series.index, series.values, marker='.', linewidth=1, label=comp)
ax2.set_xlabel("Date")
ax2.set_ylabel("Mentions")
ax2.set_title("Mentions Trajectories")
ax2.legend()
plt.xticks(rotation=40)
st.pyplot(fig2)

# Area Chart: share_of_voice stacked
st.markdown("**Share of Voice — Stacked Area**")
sov_plot = sov_pivot.copy()
# reindex to ensure continuous date index
sov_plot = sov_plot.sort_index()
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.stackplot(sov_plot.index, sov_plot.T.values, labels=sov_plot.columns)
ax3.set_xlabel("Date")
ax3.set_ylabel("Share of Voice (proportion)")
ax3.set_title("Market Share (Share of Voice) Over Time")
ax3.legend(loc='upper left')
plt.xticks(rotation=40)
st.pyplot(fig3)

# ----------------------------
# Benchmarks: normalized radar-like / bar
# ----------------------------
st.markdown("---")
st.subheader("Competitor Benchmarks")

# Compute normalized metrics
bench = filtered.groupby("competitor").agg({
    "sentiment_score": "mean",
    "mentions": "mean",
    "share_of_voice": "mean"
})
bench_norm = (bench - bench.min()) / (bench.max() - bench.min())
st.write("Normalized benchmarks (0-1) — higher is better.")
fig4, ax4 = plt.subplots(figsize=(10, 4))
bench_norm.plot(kind="bar", ax=ax4)
ax4.set_ylabel("Normalized Score")
ax4.set_title("Competitor Benchmark Comparison")
plt.xticks(rotation=45)
st.pyplot(fig4)

# ----------------------------
# Alerts: rolling drop detection + IsolationForest anomalies
# ----------------------------
st.markdown("---")
st.subheader("🚨 Alerts & Anomalies")

# Rolling-drop alerts: detect when rolling mean drops by specified pct relative to prior window
alerts = []
rolling_window = max(3, smooth_window)
for comp in sent_pivot.columns:
    s = sent_pivot[comp].sort_index().rolling(rolling_window, min_periods=1).mean()
    prev = s.shift(rolling_window)
    # percent drop: (prev - current)/abs(prev)
    pct_drop = (prev - s) / (np.abs(prev) + 1e-9) * 100
    # flag dates where drop > alert_drop_pct
    drop_dates = pct_drop[pct_drop > alert_drop_pct]
    for dt, val in drop_dates.iteritems():
        alerts.append({"date": dt, "competitor": comp, "drop_pct": float(val)})

alerts_df = pd.DataFrame(alerts).sort_values(["date", "drop_pct"], ascending=[False, False])

# IsolationForest on features (sentiment, mentions, share_of_voice) per day across competitors
# Prepare daily matrix
daily_features = filtered.groupby(["date","competitor"]).agg({
    "sentiment_score":"mean",
    "mentions":"mean",
    "share_of_voice":"mean"
}).reset_index()
# pivot to wide by competitor to run isolation forest across date rows (a simpler approach)
wide = daily_features.pivot(index="date", columns="competitor", values=["sentiment_score","mentions","share_of_voice"]).fillna(0)
# build feature matrix
X = wide.values
if len(X) > 10:
    iso = IsolationForest(contamination=anomaly_contamination, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)
    anomalies_idx = np.where(preds == -1)[0]
    anomaly_dates = wide.index[anomalies_idx]
else:
    anomaly_dates = []

# Display alerts
st.markdown("**Rolling Drop Alerts**")
if alerts_df.empty:
    st.success("No rolling-drop alerts found.")
else:
    st.table(alerts_df.head(12))

st.markdown("**Multivariate Anomalies (IsolationForest)**")
if len(anomaly_dates) == 0:
    st.success("No multivariate anomalies detected with current sensitivity.")
else:
    st.table(pd.DataFrame({"anomaly_date": anomaly_dates}).head(12))

# ----------------------------
# Forecast section: per-competitor Holt-Winters
# ----------------------------
st.markdown("---")
st.subheader("🔮 Lightweight Forecast (Holt-Winters)")

forecast_comp = st.selectbox("Select competitor to forecast", options=sent_pivot.columns)
forecast_horizon = st.slider("Forecast horizon (days)", 3, 30, 7)

series = sent_pivot[forecast_comp].sort_index()
if series.dropna().shape[0] < 6:
    st.warning("Not enough history to forecast for selected competitor.")
else:
    try:
        # fit additive seasonal model with daily frequency if enough data
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast_idx = pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=forecast_horizon)
        pred = fit.forecast(forecast_horizon)
        figf, axf = plt.subplots(figsize=(10,4))
        axf.plot(series.index, series.values, label="Historical")
        axf.plot(forecast_idx, pred.values, label="Forecast", marker='o')
        axf.set_title(f"Forecast for {forecast_comp}")
        axf.set_xlabel("Date")
        axf.set_ylabel("Sentiment Score")
        axf.legend()
        plt.xticks(rotation=40)
        st.pyplot(figf)
        st.write(pd.DataFrame({"date": forecast_idx.date, "forecast": pred.values}).reset_index(drop=True))
    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ----------------------------
# Export & Download
# ----------------------------
st.markdown("---")
st.subheader("Export & Save")
csv_buf = io.StringIO()
filtered.to_csv(csv_buf, index=False)
st.download_button("Download filtered data (CSV)", data=csv_buf.getvalue(), file_name="filtered_competitor_data.csv", mime="text/csv")

st.info("Done — use filters, explore trajectories, review alerts, and export results.")
