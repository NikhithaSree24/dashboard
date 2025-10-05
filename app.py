# app.py
"""
AI Strategic Intelligence Dashboard
Task 4 – Streamlit implementation for real-time AI market intelligence.
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

st.set_page_config(page_title="AI Strategic Intelligence Dashboard", layout="wide")

# ----------------------------
# Generate AI-focused demo data
# ----------------------------
@st.cache_data
def generate_ai_demo_data(days=240):
    """Generate demo dataset restricted to AI-related competitors/sectors"""
    competitors = [
        ("OpenAI", "Generative AI"),
        ("Google DeepMind", "AI Research"),
        ("Anthropic", "LLMs"),
        ("Meta AI", "AI Research"),
        ("Cohere", "NLP/LLMs"),
    ]
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    records = []
    for comp, sector in competitors:
        base_sent = rng.normal(0.1, 0.05)
        base_mentions = rng.integers(50, 200)
        sov_base = rng.uniform(0.05, 0.35)
        for i, d in enumerate(dates):
            trend = 0.0008 * i
            seasonal = 0.04 * np.sin(i * 2 * np.pi / 30)
            shock = rng.choice([0, 0, 0, -0.2, 0.2], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            sentiment = np.clip(base_sent + trend + seasonal + rng.normal(0, 0.03) + shock * (rng.random() < 0.02), -1, 1)
            mentions = int(base_mentions + 3 * i / 30 + rng.normal(0, 15))
            sov = max(0.01, sov_base + 0.0005 * i + 0.02 * np.sin(i * 2 * np.pi / 45) + rng.normal(0, 0.02))
            records.append({
                "date": d.date(),
                "competitor": comp,
                "sector": sector,
                "sentiment_score": float(sentiment),
                "mentions": max(0, mentions),
                "share_of_voice": float(sov)
            })
    df = pd.DataFrame.from_records(records)
    df_sum = df.groupby("date")["share_of_voice"].transform("sum")
    df["share_of_voice"] = df["share_of_voice"] / df_sum
    return df

# ----------------------------
# Page title
# ----------------------------
st.title("🤖 Strategic Intelligence Dashboard — AI Competitor Insights")
st.markdown("""
**Module 4 Objective:**  
Visualize real-time AI industry insights — track competitor sentiment, share of voice, trend evolution, and alerts.
""")

# ----------------------------
# Upload / Demo Data
# ----------------------------
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use AI Demo Data", value=True if uploaded is None else False)

if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=["date"])
        df["date"] = df["date"].dt.date
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif use_demo:
    df = generate_ai_demo_data(days=240)
else:
    st.info("Upload your dataset or enable demo data to continue.")
    st.stop()

required_cols = {"date", "competitor", "sector", "sentiment_score", "mentions", "share_of_voice"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset missing required columns. Required: {required_cols}")
    st.stop()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters & Settings")
sectors = sorted(df["sector"].unique())
selected_sector = st.sidebar.multiselect("Sector", options=sectors, default=sectors)
competitors = sorted(df[df["sector"].isin(selected_sector)]["competitor"].unique())
selected_competitors = st.sidebar.multiselect("Competitors", options=competitors, default=competitors)
date_min, date_max = df["date"].min(), df["date"].max()
start_date, end_date = st.sidebar.date_input("Date Range", [date_min, date_max])
smooth_window = st.sidebar.slider("Smoothing Window (days)", 1, 21, 7)
alert_drop_pct = st.sidebar.slider("Alert Drop Threshold (%)", 5, 50, 15)
anomaly_contamination = st.sidebar.slider("Anomaly Sensitivity", 0.001, 0.2, 0.02)

filtered = df[
    df["sector"].isin(selected_sector) &
    df["competitor"].isin(selected_competitors) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
].copy()

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ----------------------------
# Pivot metrics
# ----------------------------
def pivot_metric(metric):
    return filtered.pivot_table(index="date", columns="competitor", values=metric, aggfunc="mean").fillna(method="ffill").fillna(0)

sent_pivot = pivot_metric("sentiment_score")
mentions_pivot = pivot_metric("mentions")
sov_pivot = pivot_metric("share_of_voice")

# ----------------------------
# Leaderboard
# ----------------------------
st.subheader("🏆 AI Competitor Leaderboard")
avg_sent = filtered.groupby("competitor")["sentiment_score"].mean().sort_values(ascending=False)
top_tbl = pd.DataFrame({
    "Avg Sentiment": avg_sent.round(3),
    "Avg Mentions": filtered.groupby("competitor")["mentions"].mean().round(0),
    "Avg Share of Voice": filtered.groupby("competitor")["share_of_voice"].mean().round(3)
}).loc[selected_competitors]
st.table(top_tbl)

# ----------------------------
# Visualization 1: Sentiment Trends
# ----------------------------
st.markdown("### 📈 Sentiment Trajectories")
fig, ax = plt.subplots(figsize=(12, 4))
for comp in sent_pivot.columns:
    ax.plot(sent_pivot.index, sent_pivot[comp].rolling(smooth_window).mean(), label=comp)
ax.set_xlabel("Date")
ax.set_ylabel("Sentiment Score")
ax.legend()
plt.xticks(rotation=40)
st.pyplot(fig)

# ----------------------------
# Visualization 2: Mentions
# ----------------------------
st.markdown("### 📰 Mentions Over Time")
fig2, ax2 = plt.subplots(figsize=(12, 4))
for comp in mentions_pivot.columns:
    ax2.plot(mentions_pivot.index, mentions_pivot[comp].rolling(smooth_window).mean(), label=comp)
ax2.set_xlabel("Date")
ax2.set_ylabel("Mentions")
ax2.legend()
plt.xticks(rotation=40)
st.pyplot(fig2)

# ----------------------------
# Visualization 3: Share of Voice
# ----------------------------
st.markdown("### 🔊 Share of Voice (Stacked Area)")
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.stackplot(sov_pivot.index, sov_pivot.T.values, labels=sov_pivot.columns)
ax3.set_xlabel("Date")
ax3.set_ylabel("Share of Voice")
ax3.legend(loc="upper left")
plt.xticks(rotation=40)
st.pyplot(fig3)

# ----------------------------
# Alerts & Anomalies
# ----------------------------
st.markdown("---")
st.subheader("🚨 Alerts & Anomalies")

alerts = []
for comp, s in sent_pivot.items():
    s = s.dropna()
    prev = s.shift(1)
    pct_drop = (prev - s) / (np.abs(prev) + 1e-9) * 100
    drop_dates = pct_drop[pct_drop > alert_drop_pct]
    for dt, val in drop_dates.items():
        alerts.append({"date": dt, "competitor": comp, "drop_pct": float(val)})

alerts_df = pd.DataFrame(alerts).sort_values(["date", "drop_pct"], ascending=[True, False])
if alerts_df.empty:
    st.success("✅ No sudden sentiment drops detected.")
else:
    st.warning("⚠️ Sentiment Drop Alerts Detected:")
    st.dataframe(alerts_df.head(10))

# IsolationForest anomaly detection
daily_features = filtered.groupby(["date", "competitor"]).agg({
    "sentiment_score": "mean",
    "mentions": "mean",
    "share_of_voice": "mean"
}).reset_index()

wide = daily_features.pivot(index="date", columns="competitor",
                            values=["sentiment_score", "mentions", "share_of_voice"]).fillna(0)
X = wide.values
if len(X) > 10:
    iso = IsolationForest(contamination=anomaly_contamination, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)
    anomaly_dates = wide.index[np.where(preds == -1)[0]]
else:
    anomaly_dates = []

st.markdown("**Multivariate Anomalies:**")
if len(anomaly_dates) == 0:
    st.success("No anomalies found with current sensitivity.")
else:
    st.dataframe(pd.DataFrame({"date": anomaly_dates}).head(10))

# ----------------------------
# Forecast
# ----------------------------
st.markdown("---")
st.subheader("🔮 AI Sentiment Forecast (Holt-Winters)")
forecast_comp = st.selectbox("Select Competitor", options=sent_pivot.columns)
forecast_horizon = st.slider("Forecast Horizon (days)", 3, 30, 7)

series = sent_pivot[forecast_comp].sort_index()
if len(series.dropna()) < 6:
    st.warning("Not enough data to forecast.")
else:
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        forecast_idx = pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=forecast_horizon)
        pred = fit.forecast(forecast_horizon)
        figf, axf = plt.subplots(figsize=(10, 4))
        axf.plot(series.index, series.values, label="Historical")
        axf.plot(forecast_idx, pred.values, label="Forecast", marker="o")
        axf.set_title(f"{forecast_comp} Sentiment Forecast")
        axf.legend()
        st.pyplot(figf)
        st.dataframe(pd.DataFrame({"date": forecast_idx.date, "forecast": pred.values}))
    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ----------------------------
# Download
# ----------------------------
st.markdown("---")
st.subheader("📥 Export Filtered Data")
csv_buf = io.StringIO()
filtered.to_csv(csv_buf, index=False)
st.download_button("Download CSV", csv_buf.getvalue(), "filtered_ai_data.csv", "text/csv")

st.info("✅ Dashboard Ready — Explore trends, alerts, and forecasts for AI competitors.")
