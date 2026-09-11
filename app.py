"""
ASTRA-SCREEN
Autonomous Space-grade Thermal Reliability Assessment & Anomaly Screening
ISRO Smart India Hackathon — Problem Statement SIH 26170

AI-driven burn-in reliability screening for electronic component lots.
Predicts long-duration leakage-current drift from short-window burn-in data
and flags lots whose predicted drift trajectory exceeds a safety envelope.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ASTRA-SCREEN | ISRO SIH 26170",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants / lot configuration
# --------------------------------------------------------------------------
LOT_SIZE = 100
BURN_IN_HOURS = np.array([0, 24, 96, 168])
FULL_LIFE_HOUR = 168
COMPONENT_IDS = [f"C{str(i).zfill(4)}" for i in range(1, LOT_SIZE + 1)]

RISK_BANDS = [
    (0, 15, "LOW", "#2e7d32"),
    (15, 40, "MODERATE", "#f9a825"),
    (40, 101, "HIGH", "#c62828"),
]


# --------------------------------------------------------------------------
# Synthetic lot data generator (deterministic per component ID)
# In production this is replaced by real burn-in telemetry ingestion.
# --------------------------------------------------------------------------
@st.cache_data
def generate_lot(lot_size: int, seed: int = 42) -> pd.DataFrame:
    rows = []
    rng_master = np.random.default_rng(seed)
    base_leakage = rng_master.normal(9.8, 0.15, lot_size)
    true_slopes = rng_master.normal(0.006, 0.004, lot_size)
    # inject a handful of genuine outlier / infant-mortality units
    outlier_idx = rng_master.choice(lot_size, size=max(2, lot_size // 25), replace=False)
    true_slopes[outlier_idx] += rng_master.uniform(0.05, 0.12, len(outlier_idx))

    for i, cid in enumerate(COMPONENT_IDS[:lot_size]):
        rng = np.random.default_rng(seed + i)
        noise = rng.normal(0, 0.05, len(BURN_IN_HOURS))
        actual = base_leakage[i] + true_slopes[i] * BURN_IN_HOURS + noise
        actual[0] = base_leakage[i] + abs(noise[0]) * 0.2  # t=0 anchor
        pred_slope = true_slopes[i] + rng.normal(0, 0.002)
        predicted = actual[0] + pred_slope * BURN_IN_HOURS
        for h, a, p in zip(BURN_IN_HOURS, actual, predicted):
            rows.append(
                {
                    "component_id": cid,
                    "hour": h,
                    "actual_iddq_uA": round(a, 4),
                    "predicted_iddq_uA": round(p, 4),
                    "true_slope": true_slopes[i],
                    "pred_slope": pred_slope,
                }
            )
    return pd.DataFrame(rows)


def robust_z(values: np.ndarray, x: float) -> float:
    """Median-absolute-deviation based robust z-score (resistant to outliers)."""
    med = np.median(values)
    mad = np.median(np.abs(values - med)) or 1e-9
    return 0.6745 * (x - med) / mad


def risk_label(score: float):
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return label, color
    return "HIGH", "#c62828"


def compute_risk_score(pred_slope: float, safety_slope: float, anomaly_sigma: float) -> float:
    """
    Blend two signals into a single 0-100 risk score:
      - how far the predicted drift slope exceeds the allowed safety slope
      - how anomalous the component is relative to the rest of the lot
    """
    slope_ratio = max(0.0, (pred_slope - safety_slope) / safety_slope) if safety_slope else 0.0
    slope_component = min(70.0, slope_ratio * 100)
    anomaly_component = min(30.0, max(0.0, anomaly_sigma) * 10)
    return round(slope_component + anomaly_component, 1)


# --------------------------------------------------------------------------
# Sidebar — screening parameters
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Screening Parameters")

    safety_slope = st.slider(
        "Safety Drift Slope (µA/hour)",
        min_value=0.01,
        max_value=0.20,
        value=0.08,
        step=0.01,
        help="Maximum leakage-current drift rate a component may exhibit before "
        "being flagged as unfit for payload integration.",
    )

    st.markdown("---")
    lot_df = generate_lot(LOT_SIZE)
    component_id = st.selectbox("Select Component ID", COMPONENT_IDS, index=5)

    st.markdown("---")
    st.markdown("### 📦 Lot Summary")
    st.caption(f"Total units in lot: **{LOT_SIZE}**")
    st.caption(f"Burn-in checkpoints: **{list(BURN_IN_HOURS)} h**")

    st.markdown("---")
    st.caption("ASTRA-SCREEN · ISRO SIH 26170")
    st.caption("AI-Driven Anomaly Detection & Burn-In Reliability Screening")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.6rem;">
        <span style="font-size:2rem;">🛰️</span>
        <span style="font-size:2rem; font-weight:800;">ASTRA-SCREEN</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "##### AI-Driven Anomaly Detection & Burn-In Reliability Screening &nbsp;·&nbsp; "
    "*ISRO Smart India Hackathon — SIH 26170*"
)
st.divider()

# --------------------------------------------------------------------------
# Per-component computation
# --------------------------------------------------------------------------
comp_df = lot_df[lot_df.component_id == component_id].sort_values("hour")
pred_slope = comp_df["pred_slope"].iloc[0]
iddq_24h = comp_df.loc[comp_df.hour == 24, "actual_iddq_uA"].iloc[0]
drift_168h = comp_df.loc[comp_df.hour == 168, "predicted_iddq_uA"].iloc[0]

lot_slopes = lot_df.drop_duplicates("component_id")["pred_slope"].values
anomaly_sigma = robust_z(lot_slopes, pred_slope)
risk_score = compute_risk_score(pred_slope, safety_slope, anomaly_sigma)
label, color = risk_label(risk_score)
verdict_ok = pred_slope <= safety_slope and risk_score < 40

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Lot Size", f"{LOT_SIZE} units")
k2.metric("24h Current (Iddq)", f"{iddq_24h:.2f} µA")
k3.metric("Predicted 168h Drift", f"{drift_168h:.2f} µA")
k4.metric("Calculated Risk Score", f"{risk_score:.1f} / 100")

st.divider()

# --------------------------------------------------------------------------
# Chart + Verdict
# --------------------------------------------------------------------------
left, right = st.columns([2.1, 1])

with left:
    st.subheader("📈 Time-Series Burn-In Trajectory")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comp_df.hour,
            y=comp_df.actual_iddq_uA,
            mode="lines+markers",
            name="Actual Ground Truth",
            line=dict(color="#7fd6c2", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp_df.hour,
            y=comp_df.predicted_iddq_uA,
            mode="lines+markers",
            name="AI Predicted Trajectory",
            line=dict(color="#f4b942", width=3, dash="dash"),
        )
    )
    # safety envelope reference line from t0
    t0 = comp_df.actual_iddq_uA.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=BURN_IN_HOURS,
            y=t0 + safety_slope * BURN_IN_HOURS,
            mode="lines",
            name="Safety Envelope Limit",
            line=dict(color="#e05252", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Burn-In Hours",
        yaxis_title="Leakage Current (µA)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("✅ QA Screening Verdict")
    if verdict_ok:
        st.success(f"ACCEPTED FOR PAYLOAD  —  Risk: {label}")
    elif label == "MODERATE":
        st.warning(f"FLAGGED FOR REVIEW  —  Risk: {label}")
    else:
        st.error(f"REJECTED FOR PAYLOAD  —  Risk: {label}")

    st.markdown(
        f"""
| Metric | Value |
|---|---|
| Robust Lot Anomaly (σ) | `{anomaly_sigma:+.2f}` |
| Predicted Drift Slope | `{pred_slope:.4f} µA/h` |
| Allowed Safety Slope | `{safety_slope:.4f} µA/h` |
| Component ID | `{component_id}` |
        """
    )
    with st.expander("Why this verdict?"):
        st.write(
            f"- Predicted drift slope is "
            f"{'within' if pred_slope <= safety_slope else 'above'} the {safety_slope:.4f} µA/h safety envelope.\n"
            f"- Component sits at **{anomaly_sigma:+.2f}σ** relative to the rest of the lot "
            f"(robust, outlier-resistant estimate).\n"
            f"- Combined risk score of **{risk_score:.1f}/100** falls in the **{label}** band."
        )

st.divider()

# --------------------------------------------------------------------------
# Fleet / batch overview
# --------------------------------------------------------------------------
st.subheader("🛰️ Lot-Wide Screening Overview")

summary_rows = []
for cid, sub in lot_df.groupby("component_id"):
    s = sub.sort_values("hour")
    ps = s["pred_slope"].iloc[0]
    sig = robust_z(lot_slopes, ps)
    rs = compute_risk_score(ps, safety_slope, sig)
    lbl, _ = risk_label(rs)
    summary_rows.append(
        {
            "Component ID": cid,
            "Predicted Slope (µA/h)": round(ps, 4),
            "Anomaly (σ)": round(sig, 2),
            "Risk Score": rs,
            "Risk Band": lbl,
            "Verdict": "ACCEPTED" if (ps <= safety_slope and rs < 40) else (
                "REVIEW" if lbl == "MODERATE" else "REJECTED"
            ),
        }
    )
summary_df = pd.DataFrame(summary_rows).sort_values("Risk Score", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("✅ Accepted", int((summary_df.Verdict == "ACCEPTED").sum()))
c2.metric("🟡 Flagged for Review", int((summary_df.Verdict == "REVIEW").sum()))
c3.metric("🔴 Rejected", int((summary_df.Verdict == "REJECTED").sum()))

st.dataframe(
    summary_df.reset_index(drop=True),
    use_container_width=True,
    height=320,
)

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Full QA Report (CSV)",
    data=csv,
    file_name="astra_screen_qa_report.csv",
    mime="text/csv",
)

st.caption(
    "ASTRA-SCREEN · Synthetic demo data for illustration. Replace `generate_lot()` "
    "with a real burn-in telemetry ingestion pipeline for production screening."
)
