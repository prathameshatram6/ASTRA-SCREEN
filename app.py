"""
ASTRA-SCREEN
Autonomous Space-grade Thermal Reliability Assessment & Anomaly Screening
ISRO Smart India Hackathon — Problem Statement SIH 26170

AI-driven burn-in reliability screening for electronic component lots.
Predicts long-duration leakage-current drift from short-window burn-in data
and flags lots whose predicted drift trajectory exceeds a safety envelope.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

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
# Constants
# --------------------------------------------------------------------------
LOT_SIZE = 100
BURN_IN_HOURS = np.array([0, 24, 96, 168])
COMPONENT_IDS = [f"C{str(i).zfill(4)}" for i in range(1, LOT_SIZE + 1)]

RISK_BANDS = [
    (0, 15, "LOW", "#2e7d32"),
    (15, 40, "MODERATE", "#f9a825"),
    (40, 101, "HIGH", "#c62828"),
]

REQUIRED_COLS = {"component_id", "hour", "actual_iddq_uA"}


# --------------------------------------------------------------------------
# Synthetic lot data generator (deterministic, used when no CSV is uploaded)
# --------------------------------------------------------------------------
@st.cache_data
def generate_lot(lot_size: int, seed: int = 42) -> pd.DataFrame:
    rows = []
    rng_master = np.random.default_rng(seed)
    base_leakage = rng_master.normal(9.8, 0.15, lot_size)
    true_slopes = rng_master.normal(0.006, 0.004, lot_size)
    outlier_idx = rng_master.choice(lot_size, size=max(2, lot_size // 25), replace=False)
    true_slopes[outlier_idx] += rng_master.uniform(0.05, 0.12, len(outlier_idx))

    for i, cid in enumerate(COMPONENT_IDS[:lot_size]):
        rng = np.random.default_rng(seed + i)
        noise = rng.normal(0, 0.05, len(BURN_IN_HOURS))
        actual = base_leakage[i] + true_slopes[i] * BURN_IN_HOURS + noise
        actual[0] = base_leakage[i] + abs(noise[0]) * 0.2
        pred_slope = true_slopes[i] + rng.normal(0, 0.002)
        predicted = actual[0] + pred_slope * BURN_IN_HOURS
        for h, a, p in zip(BURN_IN_HOURS, actual, predicted):
            rows.append(
                {
                    "component_id": cid,
                    "hour": h,
                    "actual_iddq_uA": round(a, 4),
                    "predicted_iddq_uA": round(p, 4),
                }
            )
    return pd.DataFrame(rows)


def fit_predicted_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every component has a 'predicted_iddq_uA' column.
    If missing (e.g. a raw CSV upload with only ground-truth readings),
    fit a per-component linear drift model (least squares) and use it
    to generate the predicted trajectory.
    """
    if "predicted_iddq_uA" in df.columns and df["predicted_iddq_uA"].notna().all():
        return df

    out = []
    for cid, sub in df.groupby("component_id"):
        sub = sub.sort_values("hour").copy()
        if len(sub) >= 2:
            slope, intercept = np.polyfit(sub["hour"], sub["actual_iddq_uA"], 1)
        else:
            slope, intercept = 0.0, sub["actual_iddq_uA"].iloc[0]
        sub["predicted_iddq_uA"] = (intercept + slope * sub["hour"]).round(4)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def compute_pred_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a single predicted-drift-slope value per component."""
    slopes = []
    for cid, sub in df.groupby("component_id"):
        sub = sub.sort_values("hour")
        if len(sub) >= 2:
            s = np.polyfit(sub["hour"], sub["predicted_iddq_uA"], 1)[0]
        else:
            s = 0.0
        slopes.append({"component_id": cid, "pred_slope": s})
    return pd.DataFrame(slopes)


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
    slope_ratio = max(0.0, (pred_slope - safety_slope) / safety_slope) if safety_slope else 0.0
    slope_component = min(70.0, slope_ratio * 100)
    anomaly_component = min(30.0, max(0.0, anomaly_sigma) * 10)
    return round(slope_component + anomaly_component, 1)


def verdict_for(pred_slope, safety_slope, risk_score, label):
    if pred_slope <= safety_slope and risk_score < 40:
        return "ACCEPTED"
    return "REVIEW" if label == "MODERATE" else "REJECTED"


@st.cache_data
def build_summary(lot_df: pd.DataFrame, safety_slope: float) -> pd.DataFrame:
    slope_df = compute_pred_slopes(lot_df)
    lot_slopes = slope_df["pred_slope"].values
    rows = []
    for _, r in slope_df.iterrows():
        sig = robust_z(lot_slopes, r.pred_slope)
        rs = compute_risk_score(r.pred_slope, safety_slope, sig)
        lbl, _ = risk_label(rs)
        rows.append(
            {
                "Component ID": r.component_id,
                "Predicted Slope (µA/h)": round(r.pred_slope, 4),
                "Anomaly (σ)": round(sig, 2),
                "Risk Score": rs,
                "Risk Band": lbl,
                "Verdict": verdict_for(r.pred_slope, safety_slope, rs, lbl),
            }
        )
    return pd.DataFrame(rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)


def make_pdf_certificate(component_id, risk_score, label, pred_slope, safety_slope,
                          anomaly_sigma, verdict, comp_df) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 30, 60)
    pdf.cell(0, 12, "ASTRA-SCREEN QA Certificate", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, "AI-Driven Anomaly Detection & Burn-In Reliability Screening", ln=True)
    pdf.cell(0, 7, "ISRO Smart India Hackathon - SIH 26170", ln=True)
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Component ID: {component_id}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    verdict_color = {"ACCEPTED": (46, 125, 50), "REVIEW": (249, 168, 37), "REJECTED": (198, 40, 40)}
    r, g, b = verdict_color.get(verdict, (0, 0, 0))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  VERDICT: {verdict}  ", ln=True, fill=True)
    pdf.ln(6)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    rows = [
        ("Risk Score", f"{risk_score:.1f} / 100 ({label})"),
        ("Predicted Drift Slope", f"{pred_slope:.4f} uA/h"),
        ("Allowed Safety Slope", f"{safety_slope:.4f} uA/h"),
        ("Robust Lot Anomaly (sigma)", f"{anomaly_sigma:+.2f}"),
    ]
    for k, v in rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 8, k, border=0)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(v), ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Burn-In Readings", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(40, 7, "Hour", border=1, fill=True)
    pdf.cell(60, 7, "Actual (uA)", border=1, fill=True)
    pdf.cell(60, 7, "Predicted (uA)", border=1, fill=True, ln=True)
    pdf.set_font("Helvetica", "", 10)
    for _, row in comp_df.sort_values("hour").iterrows():
        pdf.cell(40, 7, str(int(row.hour)), border=1)
        pdf.cell(60, 7, f"{row.actual_iddq_uA:.4f}", border=1)
        pdf.cell(60, 7, f"{row.predicted_iddq_uA:.4f}", border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "This certificate is generated by the ASTRA-SCREEN automated reliability "
        "screening pipeline and reflects predicted drift behaviour based on burn-in "
        "telemetry available at the time of generation.",
    )
    return bytes(pdf.output(dest="S"))


# --------------------------------------------------------------------------
# Sidebar — data source + screening parameters
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📥 Data Source")
    uploaded = st.file_uploader(
        "Upload burn-in CSV (optional)",
        type=["csv"],
        help="Columns required: component_id, hour, actual_iddq_uA. "
        "predicted_iddq_uA is optional — if omitted, ASTRA-SCREEN fits a "
        "per-component linear drift model automatically.",
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            missing = REQUIRED_COLS - set(raw_df.columns)
            if missing:
                st.error(f"CSV is missing required column(s): {', '.join(missing)}")
                st.stop()
            lot_df = fit_predicted_trajectory(raw_df)
            st.success(f"Loaded {lot_df.component_id.nunique()} components from upload.")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()
    else:
        lot_df = generate_lot(LOT_SIZE)
        st.caption("Using synthetic demo data. Upload a CSV to screen real telemetry.")

    all_ids = sorted(lot_df.component_id.unique().tolist())

    st.markdown("---")
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
    component_id = st.selectbox("Select Component ID", all_ids, index=min(5, len(all_ids) - 1))

    st.markdown("---")
    st.markdown("### 📦 Lot Summary")
    st.caption(f"Total units in lot: **{lot_df.component_id.nunique()}**")
    st.caption(f"Burn-in checkpoints: **{sorted(lot_df.hour.unique().tolist())} h**")

    st.markdown("---")
    st.caption("ASTRA-SCREEN · ISRO SIH 26170")

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
# Shared computations
# --------------------------------------------------------------------------
summary_df = build_summary(lot_df, safety_slope)
slope_df = compute_pred_slopes(lot_df)
lot_slopes = slope_df["pred_slope"].values

comp_df = lot_df[lot_df.component_id == component_id].sort_values("hour")
pred_slope = float(slope_df.loc[slope_df.component_id == component_id, "pred_slope"].iloc[0])
iddq_24h_row = comp_df.loc[comp_df.hour == 24, "actual_iddq_uA"]
iddq_24h = float(iddq_24h_row.iloc[0]) if len(iddq_24h_row) else float(comp_df.actual_iddq_uA.iloc[-1])
drift_168h_row = comp_df.loc[comp_df.hour == comp_df.hour.max(), "predicted_iddq_uA"]
drift_168h = float(drift_168h_row.iloc[0])

anomaly_sigma = robust_z(lot_slopes, pred_slope)
risk_score = compute_risk_score(pred_slope, safety_slope, anomaly_sigma)
label, color = risk_label(risk_score)
verdict = verdict_for(pred_slope, safety_slope, risk_score, label)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_screen, tab_analytics, tab_compare, tab_export = st.tabs(
    ["🔍 Component Screening", "📊 Lot Analytics", "🆚 Compare Components", "📄 Export Certificate"]
)

# ============================== TAB 1: SCREENING ==========================
with tab_screen:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Lot Size", f"{lot_df.component_id.nunique()} units")
    k2.metric("24h Current (Iddq)", f"{iddq_24h:.2f} µA")
    k3.metric("Predicted Final Drift", f"{drift_168h:.2f} µA")
    k4.metric("Calculated Risk Score", f"{risk_score:.1f} / 100")

    st.divider()
    left, right = st.columns([2.1, 1])

    with left:
        st.subheader("📈 Time-Series Burn-In Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comp_df.hour, y=comp_df.actual_iddq_uA, mode="lines+markers",
            name="Actual Ground Truth", line=dict(color="#7fd6c2", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=comp_df.hour, y=comp_df.predicted_iddq_uA, mode="lines+markers",
            name="AI Predicted Trajectory", line=dict(color="#f4b942", width=3, dash="dash"),
        ))
        t0 = comp_df.actual_iddq_uA.iloc[0]
        hrs = comp_df.hour.values
        fig.add_trace(go.Scatter(
            x=hrs, y=t0 + safety_slope * hrs, mode="lines",
            name="Safety Envelope Limit", line=dict(color="#e05252", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            template="plotly_dark", height=430, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Burn-In Hours", yaxis_title="Leakage Current (µA)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("✅ QA Screening Verdict")
        if verdict == "ACCEPTED":
            st.success(f"ACCEPTED FOR PAYLOAD  —  Risk: {label}")
        elif verdict == "REVIEW":
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

# ============================== TAB 2: ANALYTICS ===========================
with tab_analytics:
    st.subheader("🛰️ Lot-Wide Screening Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Accepted", int((summary_df.Verdict == "ACCEPTED").sum()))
    c2.metric("🟡 Flagged for Review", int((summary_df.Verdict == "REVIEW").sum()))
    c3.metric("🔴 Rejected", int((summary_df.Verdict == "REJECTED").sum()))

    st.markdown("#### Risk Score Distribution")
    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=summary_df["Risk Score"], nbinsx=20, marker_color="#f4b942", opacity=0.85,
    ))
    hist.add_vline(x=15, line_dash="dot", line_color="#2e7d32", annotation_text="Low/Moderate")
    hist.add_vline(x=40, line_dash="dot", line_color="#c62828", annotation_text="Moderate/High")
    hist.update_layout(
        template="plotly_dark", height=280, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Risk Score", yaxis_title="Number of Components",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(hist, use_container_width=True)

    st.markdown("#### Filter & Search")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        band_filter = st.multiselect(
            "Risk band", options=["LOW", "MODERATE", "HIGH"],
            default=["LOW", "MODERATE", "HIGH"],
        )
    with fc2:
        search = st.text_input("Search Component ID", placeholder="e.g. C0006")

    filtered = summary_df[summary_df["Risk Band"].isin(band_filter)]
    if search:
        filtered = filtered[filtered["Component ID"].str.contains(search, case=False, na=False)]

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=320)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full QA Report (CSV)", data=csv_bytes,
        file_name="astra_screen_qa_report.csv", mime="text/csv",
    )

# ============================== TAB 3: COMPARE =============================
with tab_compare:
    st.subheader("🆚 Compare Multiple Components")
    default_sel = all_ids[:3] if len(all_ids) >= 3 else all_ids
    compare_ids = st.multiselect(
        "Select components to overlay (up to 8)", options=all_ids,
        default=default_sel, max_selections=8,
    )

    if compare_ids:
        cfig = go.Figure()
        palette = ["#f4b942", "#7fd6c2", "#e05252", "#8ab4f8", "#c792ea",
                   "#ffab91", "#a5d6a7", "#ce93d8"]
        for i, cid in enumerate(compare_ids):
            sub = lot_df[lot_df.component_id == cid].sort_values("hour")
            cfig.add_trace(go.Scatter(
                x=sub.hour, y=sub.actual_iddq_uA, mode="lines+markers",
                name=cid, line=dict(color=palette[i % len(palette)], width=2.5),
            ))
        cfig.update_layout(
            template="plotly_dark", height=430, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Burn-In Hours", yaxis_title="Actual Leakage Current (µA)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(cfig, use_container_width=True)

        compare_summary = summary_df[summary_df["Component ID"].isin(compare_ids)]
        st.dataframe(compare_summary.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Select at least one component above to compare trajectories.")

# ============================== TAB 4: EXPORT ===============================
with tab_export:
    st.subheader("📄 Downloadable QA Certificate")
    st.write(
        f"Generate a one-page PDF certificate for **{component_id}** summarising its "
        "verdict, risk score, and burn-in readings — suitable for attaching to a lot "
        "acceptance record or sharing with reviewers."
    )
    pdf_bytes = make_pdf_certificate(
        component_id, risk_score, label, pred_slope, safety_slope,
        anomaly_sigma, verdict, comp_df,
    )
    st.download_button(
        "⬇️ Download QA Certificate (PDF)",
        data=pdf_bytes,
        file_name=f"astra_screen_certificate_{component_id}.pdf",
        mime="application/pdf",
    )
    st.caption("Change the selected component in the sidebar to generate a certificate for a different unit.")

st.divider()
st.caption(
    "ASTRA-SCREEN · Upload a CSV in the sidebar to screen real burn-in telemetry, "
    "or leave it empty to explore the synthetic demo lot."
)
