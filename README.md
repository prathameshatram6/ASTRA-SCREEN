# 🛰️ ASTRA-SCREEN

**AI-Driven Anomaly Detection & Burn-In Reliability Screening**
Smart India Hackathon — Problem Statement **SIH 26170** (ISRO)

ASTRA-SCREEN predicts long-duration leakage-current (Iddq) drift for
space-grade electronic components from short-window burn-in telemetry,
and screens each lot against a configurable safety envelope before it
is cleared for payload integration.

## ✨ What's new in this version

- **Robust anomaly scoring** — uses a median-absolute-deviation (MAD)
  based z-score instead of a plain mean/std, so a couple of extreme
  outliers in the lot no longer distort every other component's score.
- **Transparent risk score** — the 0–100 risk score is an explicit blend
  of (a) how far the predicted drift slope exceeds the safety envelope
  and (b) the component's anomaly position within the lot, with the
  math shown in an expandable "Why this verdict?" panel.
- **Safety envelope drawn on the chart** — the allowed drift limit is
  now plotted directly alongside actual vs. predicted trajectories.
- **Three-way verdict** — ACCEPTED / FLAGGED FOR REVIEW / REJECTED,
  instead of a binary pass/fail, so borderline components surface for
  human review rather than being silently accepted or rejected.
- **Lot-wide overview** — a new section screens all 100 units at once,
  with accepted/review/rejected counts and a sortable table.
- **CSV export** — download the full QA report for the lot in one click.

## 🚀 Running locally

```bash
git clone <your-repo-url>
cd astra-screen
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📁 Project structure

```
astra-screen/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .streamlit/config.toml   # Dark theme configuration
└── README.md
```

## 🔧 Plugging in real data

`generate_lot()` in `app.py` currently produces deterministic synthetic
burn-in data for demo purposes. Replace it with a loader that reads your
actual burn-in telemetry (e.g. from a CSV, database, or lab instrument
export) with columns:

| column | description |
|---|---|
| `component_id` | unique identifier per unit |
| `hour` | burn-in checkpoint hour (0, 24, 96, 168, ...) |
| `actual_iddq_uA` | measured leakage current |
| `predicted_iddq_uA` | model-predicted leakage current at that hour |

The rest of the scoring, charting, and verdict pipeline works unchanged.

## 📤 Pushing this to GitHub

```bash
cd astra-screen
git init
git add .
git commit -m "ASTRA-SCREEN: burn-in reliability screening dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/astra-screen.git
git push -u origin main
```

## 🖥️ Deploying

Works out of the box on **Streamlit Community Cloud**: point it at your
GitHub repo, `app.py` as the entry point, and it will pick up
`requirements.txt` automatically.
