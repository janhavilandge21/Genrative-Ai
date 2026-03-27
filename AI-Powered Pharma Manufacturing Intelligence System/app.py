"""
frontend/app.py — Streamlit UI for Pharma AI Manufacturing Intelligence System.
Run from project root: streamlit run frontend/app.py
"""

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.preprocessor import preprocess_single
from src.model_trainer import predict_batch, get_shap_values
from src.groq_advisor import analyze_batch_with_groq, get_realtime_alert
from src.simulator import generate_random_batch, stream_batches

MODELS_DIR = ROOT / "models"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma AI Intelligence System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #050709;
    --surface: #0d1117;
    --surface2: #161b22;
    --surface3: #21262d;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --text: #e6edf3;
    --muted: #8b949e;
    --border: #30363d;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 500;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface3) !important;
    color: var(--text) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
.stNumberInput input, .stSelectbox select {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.status-pass {
    background: #0d2a13;
    border: 2px solid var(--green);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.status-fail {
    background: #2d0d0d;
    border: 2px solid var(--red);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.status-warn {
    background: #2d1f00;
    border: 2px solid var(--yellow);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.ai-box {
    background: linear-gradient(135deg, #0d1a2d, #0d2040);
    border: 1px solid #1f3a6e;
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
}
.sim-row-critical { background: #2d0d0d; border-left: 4px solid #f85149; border-radius: 6px; padding: 8px 12px; margin: 4px 0; }
.sim-row-warning  { background: #2d1f00; border-left: 4px solid #d29922; border-radius: 6px; padding: 8px 12px; margin: 4px 0; }
.sim-row-normal   { background: #0d2a13; border-left: 4px solid #3fb950; border-radius: 6px; padding: 8px 12px; margin: 4px 0; }
.section-header {
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--blue);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 16px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "sim_results" not in st.session_state:
    st.session_state.sim_results = []
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ── Check models exist ─────────────────────────────────────────────────────────
def models_ready() -> bool:
    return all([
        (MODELS_DIR / "xgboost_model.pkl").exists(),
        (MODELS_DIR / "isolation_forest.pkl").exists(),
        (MODELS_DIR / "scaler.pkl").exists(),
    ])


# ── Gauge chart ────────────────────────────────────────────────────────────────
def failure_gauge(probability: float) -> go.Figure:
    color = "#f85149" if probability > 0.6 else "#d29922" if probability > 0.35 else "#3fb950"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={"reference": 30, "increasing": {"color": "#f85149"}, "decreasing": {"color": "#3fb950"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 35],  "color": "#0d2a13"},
                {"range": [35, 65], "color": "#2d1f00"},
                {"range": [65, 100],"color": "#2d0d0d"},
            ],
            "threshold": {
                "line":  {"color": "#f85149", "width": 3},
                "thickness": 0.8,
                "value": 65,
            },
        },
        title={"text": "Batch Failure Probability", "font": {"color": "#8b949e", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        font={"color": "#e6edf3"},
    )
    return fig


# ── SHAP bar chart ─────────────────────────────────────────────────────────────
def shap_chart(shap_dict: dict) -> go.Figure:
    items = list(shap_dict.items())[:10]
    features = [i[0].replace("_", " ").title() for i in items]
    values   = [i[1] for i in items]
    colors   = ["#f85149" if v > 0 else "#3fb950" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#e6edf3", "size": 11},
    ))
    fig.update_layout(
        title={"text": "Feature Impact on Failure Prediction (SHAP)", "font": {"color": "#8b949e", "size": 13}},
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis={"title": "SHAP Value (red=increases failure risk)", "color": "#8b949e", "gridcolor": "#21262d"},
        yaxis={"color": "#8b949e"},
        height=350,
        margin=dict(l=10, r=60, t=40, b=40),
        font={"color": "#e6edf3"},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:10px 0 20px 0">
  <div style="font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#58a6ff,#bc8cff,#3fb950);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    💊 Pharma AI Manufacturing Intelligence System
  </div>
  <div style="color:#8b949e;font-size:1rem;margin-top:6px">
    Batch Failure Prediction · Anomaly Detection · AI-Powered Recommendations · Real-Time Simulation
  </div>
</div>
""", unsafe_allow_html=True)


# ── Model readiness check ──────────────────────────────────────────────────────
if not models_ready():
    st.error("""
    ❌ **Models not trained yet!**

    Run the training pipeline first:
    ```
    python train_pipeline.py
    ```
    Then reload this page.
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — API Key + Model Metrics
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    groq_ok = bool(groq_key) and groq_key != "gsk_your_groq_key_here"
    st.markdown(f"Groq Status: {'✅ Connected' if groq_ok else '❌ Not configured'}")

    st.divider()

    # Model metrics
    st.markdown("### 📊 Model Performance")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        col1, col2 = st.columns(2)
        col1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
        col2.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
        col1.metric("Precision", f"{m['precision']:.3f}")
        col2.metric("Recall",    f"{m['recall']:.3f}")

    st.divider()

    # Feature importance ranking
    st.markdown("### 🔑 Top Risk Factors")
    fi_path = MODELS_DIR / "feature_importances.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)
        for i, (feat, imp) in enumerate(list(fi.items())[:8], 1):
            bar_w = int(imp * 120)
            st.markdown(
                f"<div style='font-size:0.78rem;margin:3px 0'>"
                f"<span style='color:#8b949e'>{i}.</span> "
                f"<span style='color:#e6edf3'>{feat.replace('_',' ').title()}</span>"
                f"<div style='height:5px;width:{bar_w}px;background:#58a6ff;border-radius:3px;margin-top:2px'></div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Batch Analyzer",
    "📡 Real-Time Simulation",
    "📊 Data Explorer",
    "📖 Model Insights",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: BATCH ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">⚗️ Enter Batch Parameters</p>', unsafe_allow_html=True)

    # Quick fill buttons
    qc1, qc2, qc3 = st.columns(3)
    fill_normal  = qc1.button("📋 Fill Normal Batch",   use_container_width=True)
    fill_risky   = qc2.button("⚠️ Fill Risky Batch",    use_container_width=True)
    fill_failure = qc3.button("❌ Fill Failing Batch",   use_container_width=True)

    # Default values
    defaults = {
        "normal":  dict(temp=65, press=2.5, mix_t=45, mix_rpm=120, humid=45, ph=6.8,
                        part=200, api=98, moist=2.5, gran=30, dry=55, coat=150,
                        hard=80, diss=87, exp=6, shift=0, eq=4, grade=1),
        "risky":   dict(temp=78, press=1.8, mix_t=28, mix_rpm=80, humid=62, ph=5.8,
                        part=330, api=95, moist=4.2, gran=15, dry=45, coat=105,
                        hard=58, diss=72, exp=2, shift=1, eq=9, grade=0),
        "failure": dict(temp=90, press=1.2, mix_t=15, mix_rpm=55, humid=75, ph=5.0,
                        part=380, api=92, moist=5.5, gran=9,  dry=38, coat=85,
                        hard=42, diss=55, exp=1, shift=2, eq=13, grade=0),
    }

    if fill_failure:
        d = defaults["failure"]
    elif fill_risky:
        d = defaults["risky"]
    else:
        d = defaults["normal"]

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("batch_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌡️ Thermal Parameters**")
            temperature_c   = st.number_input("Temperature (°C)",        min_value=30.0,  max_value=100.0, value=float(d["temp"]),  step=0.5, format="%.1f")
            drying_temp_c   = st.number_input("Drying Temperature (°C)", min_value=30.0,  max_value=90.0,  value=float(d["dry"]),   step=0.5, format="%.1f")

            st.markdown("**💧 Process Conditions**")
            pressure_bar    = st.number_input("Pressure (bar)",           min_value=0.5,   max_value=6.0,   value=float(d["press"]), step=0.1, format="%.2f")
            humidity_pct    = st.number_input("Humidity (%)",             min_value=10.0,  max_value=90.0,  value=float(d["humid"]), step=1.0, format="%.1f")
            ph_level        = st.number_input("pH Level",                 min_value=3.0,   max_value=10.0,  value=float(d["ph"]),    step=0.1, format="%.2f")

        with col2:
            st.markdown("**⚙️ Mixing & Granulation**")
            mixing_time_min    = st.number_input("Mixing Time (min)",        min_value=5.0,   max_value=120.0, value=float(d["mix_t"]),   step=1.0, format="%.1f")
            mixing_speed_rpm   = st.number_input("Mixing Speed (RPM)",       min_value=20.0,  max_value=300.0, value=float(d["mix_rpm"]), step=5.0, format="%.0f")
            granulation_time_min = st.number_input("Granulation Time (min)", min_value=5.0,   max_value=80.0,  value=float(d["gran"]),    step=1.0, format="%.1f")
            particle_size_um   = st.number_input("Particle Size (μm)",       min_value=50.0,  max_value=500.0, value=float(d["part"]),    step=5.0, format="%.1f")

            st.markdown("**💊 Product Quality**")
            coating_thickness_um = st.number_input("Coating Thickness (μm)", min_value=50.0,  max_value=300.0, value=float(d["coat"]),  step=5.0, format="%.1f")
            tablet_hardness_n  = st.number_input("Tablet Hardness (N)",      min_value=20.0,  max_value=160.0, value=float(d["hard"]),  step=1.0, format="%.1f")

        with col3:
            st.markdown("**🧪 Chemical Parameters**")
            active_ingredient_pct = st.number_input("Active Ingredient (%)",   min_value=85.0,  max_value=105.0, value=float(d["api"]),   step=0.1, format="%.2f")
            moisture_content_pct  = st.number_input("Moisture Content (%)",    min_value=0.1,   max_value=8.0,   value=float(d["moist"]), step=0.1, format="%.2f")
            dissolution_rate_pct  = st.number_input("Dissolution Rate (%)",    min_value=30.0,  max_value=100.0, value=float(d["diss"]),  step=0.5, format="%.1f")

            st.markdown("**👷 Operational Factors**")
            operator_experience_yr = st.number_input("Operator Experience (yr)", min_value=0.5, max_value=25.0, value=float(d["exp"]),   step=0.5, format="%.1f")
            shift = st.selectbox("Shift", options=[0, 1, 2], format_func=lambda x: ["Day (0)", "Evening (1)", "Night (2)"][x], index=d["shift"])
            equipment_age_yr = st.number_input("Equipment Age (yr)",            min_value=0.5,  max_value=20.0,  value=float(d["eq"]),    step=0.5, format="%.1f")
            raw_material_grade = st.selectbox("Raw Material Grade", options=[0, 1], format_func=lambda x: ["Standard (0)", "Premium (1)"][x], index=d["grade"])

        submitted = st.form_submit_button("🔬 Analyze Batch", use_container_width=True)

    # ── Run analysis ───────────────────────────────────────────────────────────
    if submitted:
        batch_data = {
            "temperature_c": temperature_c, "pressure_bar": pressure_bar,
            "mixing_time_min": mixing_time_min, "mixing_speed_rpm": int(mixing_speed_rpm),
            "humidity_pct": humidity_pct, "ph_level": ph_level,
            "particle_size_um": particle_size_um, "active_ingredient_pct": active_ingredient_pct,
            "moisture_content_pct": moisture_content_pct, "granulation_time_min": granulation_time_min,
            "drying_temp_c": drying_temp_c, "coating_thickness_um": coating_thickness_um,
            "tablet_hardness_n": tablet_hardness_n, "dissolution_rate_pct": dissolution_rate_pct,
            "operator_experience_yr": operator_experience_yr, "shift": shift,
            "equipment_age_yr": equipment_age_yr, "raw_material_grade": raw_material_grade,
        }

        with st.spinner("🤖 Running ML pipeline..."):
            X_scaled, features = preprocess_single(batch_data)
            prediction  = predict_batch(X_scaled)
            shap_values = get_shap_values(X_scaled, features)

        st.divider()
        st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)

        # ── Results row ───────────────────────────────────────────────────────
        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            fig_gauge = failure_gauge(prediction["failure_probability"])
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r2:
            fp = prediction["failure_probability"]
            if prediction["failure_predicted"]:
                st.markdown(f'<div class="status-fail"><div style="font-size:2.5rem">❌</div><div style="font-size:1.4rem;font-weight:700;color:#f85149">BATCH FAIL</div><div style="color:#8b949e;font-size:0.9rem">Probability: {fp*100:.1f}%</div></div>', unsafe_allow_html=True)
            elif fp > 0.40:
                st.markdown(f'<div class="status-warn"><div style="font-size:2.5rem">⚠️</div><div style="font-size:1.4rem;font-weight:700;color:#d29922">AT RISK</div><div style="color:#8b949e;font-size:0.9rem">Probability: {fp*100:.1f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-pass"><div style="font-size:2.5rem">✅</div><div style="font-size:1.4rem;font-weight:700;color:#3fb950">BATCH PASS</div><div style="color:#8b949e;font-size:0.9rem">Probability: {fp*100:.1f}%</div></div>', unsafe_allow_html=True)

        with r3:
            if prediction["is_anomaly"]:
                st.markdown(f'<div class="status-fail"><div style="font-size:2.5rem">🚨</div><div style="font-size:1.2rem;font-weight:700;color:#f85149">ANOMALY DETECTED</div><div style="color:#8b949e;font-size:0.85rem">Score: {prediction["anomaly_score"]:.4f}</div><div style="color:#8b949e;font-size:0.8rem;margin-top:4px">Unusual parameter combination</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-pass"><div style="font-size:2.5rem">🟢</div><div style="font-size:1.2rem;font-weight:700;color:#3fb950">NORMAL BATCH</div><div style="color:#8b949e;font-size:0.85rem">Score: {prediction["anomaly_score"]:.4f}</div><div style="color:#8b949e;font-size:0.8rem;margin-top:4px">No anomaly detected</div></div>', unsafe_allow_html=True)

        # ── SHAP chart ────────────────────────────────────────────────────────
        st.plotly_chart(shap_chart(shap_values), use_container_width=True)

        # ── AI Recommendation ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">🤖 AI-Powered Recommendation (Groq LLaMA3)</p>', unsafe_allow_html=True)

        if groq_ok:
            with st.spinner("🧠 Generating AI analysis via Groq..."):
                ai_response = analyze_batch_with_groq(batch_data, prediction, shap_values)
            st.markdown(f'<div class="ai-box">{ai_response.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Add your Groq API key in the sidebar to enable AI recommendations.")
            st.info("Get a free key at: https://console.groq.com")

        st.session_state.analysis_result = {
            "batch_data": batch_data,
            "prediction": prediction,
            "shap_values": shap_values,
        }


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: REAL-TIME SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📡 Real-Time Batch Stream Simulation</p>', unsafe_allow_html=True)
    st.markdown("Simulates incoming batches from the production line and reacts dynamically to each batch.")

    sc1, sc2, sc3 = st.columns(3)
    n_batches = sc1.slider("Number of batches to simulate", 5, 20, 8)
    delay_sec = sc2.slider("Delay between batches (seconds)", 0.5, 3.0, 1.0, 0.5)
    sc3.markdown("<br>", unsafe_allow_html=True)
    start_sim = sc3.button("▶️ Start Simulation", use_container_width=True)
    clear_sim = sc3.button("🗑 Clear Results",    use_container_width=True)

    if clear_sim:
        st.session_state.sim_results = []
        st.rerun()

    # Summary metrics placeholders
    m1, m2, m3, m4 = st.columns(4)
    ph_total    = m1.empty()
    ph_passed   = m2.empty()
    ph_failed   = m3.empty()
    ph_anomaly  = m4.empty()

    # Live stream placeholder
    stream_ph   = st.empty()
    chart_ph    = st.empty()

    def render_stream(results: list):
        # Summary
        total   = len(results)
        failed  = sum(1 for r in results if r["prediction"]["failure_predicted"])
        anomaly = sum(1 for r in results if r["prediction"]["is_anomaly"])
        passed  = total - failed

        ph_total.metric("📦 Total Batches", total)
        ph_passed.metric("✅ Passed", passed)
        ph_failed.metric("❌ Failed", failed)
        ph_anomaly.metric("🚨 Anomalies", anomaly)

        # Stream log
        rows_html = ""
        for r in reversed(results[-15:]):
            p   = r["prediction"]
            fp  = p["failure_probability"] * 100
            cls = f"sim-row-{r['status'].lower()}"
            icon = "❌" if r["status"] == "CRITICAL" else "⚠️" if r["status"] == "WARNING" else "✅"
            rows_html += (
                f'<div class="{cls}">'
                f'<span style="font-family:monospace;color:#8b949e">[{r["timestamp"]}]</span> '
                f'{icon} <b>{r["batch_data"]["batch_id"]}</b> &nbsp;|&nbsp; '
                f'Failure: <b style="color:{"#f85149" if fp>60 else "#d29922" if fp>35 else "#3fb950"}">{fp:.1f}%</b> &nbsp;|&nbsp; '
                f'Status: <b>{r["status"]}</b>'
                f'{"  🚨 ANOMALY" if p["is_anomaly"] else ""}'
                f'</div>'
            )
        stream_ph.markdown(rows_html, unsafe_allow_html=True)

        # Line chart of failure probability
        if len(results) > 1:
            df_sim = pd.DataFrame([
                {"Batch": r["batch_data"]["batch_id"],
                 "Failure %": r["prediction"]["failure_probability"] * 100,
                 "Status": r["status"]}
                for r in results
            ])
            fig = px.line(
                df_sim, x="Batch", y="Failure %",
                markers=True,
                color_discrete_sequence=["#58a6ff"],
                title="Live Failure Probability Stream",
            )
            fig.add_hline(y=65, line_dash="dash", line_color="#f85149",
                          annotation_text="Critical threshold (65%)")
            fig.add_hline(y=35, line_dash="dot",  line_color="#d29922",
                          annotation_text="Warning threshold (35%)")
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font={"color": "#e6edf3"}, height=280,
                xaxis={"tickangle": -30, "gridcolor": "#21262d"},
                yaxis={"range": [0, 100], "gridcolor": "#21262d"},
                margin=dict(l=40, r=20, t=40, b=60),
            )
            chart_ph.plotly_chart(fig, use_container_width=True)

    # Show existing results
    if st.session_state.sim_results:
        render_stream(st.session_state.sim_results)

    # Run simulation
    if start_sim:
        st.session_state.sim_results = []
        for result in stream_batches(n_batches=n_batches, delay_seconds=delay_sec):
            st.session_state.sim_results.append(result)
            render_stream(st.session_state.sim_results)

            # Real-time AI alert for critical batches
            if result["status"] == "CRITICAL" and groq_ok:
                alert = get_realtime_alert(result["batch_data"], result["status"])
                st.markdown(
                    f'<div style="background:#2d0d0d;border:1px solid #f85149;border-radius:8px;'
                    f'padding:10px 14px;margin:6px 0;color:#f85149">🚨 <b>AI Alert:</b> {alert}</div>',
                    unsafe_allow_html=True
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">📂 Training Dataset Explorer</p>', unsafe_allow_html=True)
    data_path = ROOT / "data" / "pharma_batches.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        total = len(df)
        failed = df["batch_failed"].sum()
        passed = total - failed

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Batches", total)
        d2.metric("✅ Passed", int(passed))
        d3.metric("❌ Failed", int(failed))
        d4.metric("Failure Rate", f"{failed/total*100:.1f}%")

        # Distribution chart
        num_cols = ["temperature_c", "moisture_content_pct", "ph_level",
                    "active_ingredient_pct", "dissolution_rate_pct", "tablet_hardness_n"]

        sel_col = st.selectbox("Select parameter to visualize:", num_cols)

        fig_dist = px.histogram(
            df, x=sel_col, color="batch_failed",
            barmode="overlay", nbins=40,
            color_discrete_map={0: "#3fb950", 1: "#f85149"},
            labels={"batch_failed": "Batch Failed", sel_col: sel_col.replace("_", " ").title()},
            title=f"Distribution of {sel_col.replace('_',' ').title()} by Batch Outcome",
        )
        fig_dist.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font={"color": "#e6edf3"}, height=320,
            xaxis={"gridcolor": "#21262d"}, yaxis={"gridcolor": "#21262d"},
            legend={"title": "0=Passed, 1=Failed"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Correlation heatmap
        if st.checkbox("Show Feature Correlation Heatmap"):
            corr = df[num_cols + ["batch_failed"]].corr()
            fig_heat = px.imshow(
                corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Feature Correlation Matrix",
            )
            fig_heat.update_layout(paper_bgcolor="#0d1117", font={"color": "#e6edf3"}, height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Raw Data Sample (first 50 rows)**")
        st.dataframe(df.head(50), use_container_width=True, height=300)
    else:
        st.warning("Dataset not found. Run `python train_pipeline.py` first.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">🤖 Model Architecture & Insights</p>', unsafe_allow_html=True)

    mi1, mi2 = st.columns(2)

    with mi1:
        st.markdown("**XGBoost Failure Classifier**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | XGBoost (Gradient Boosting) |
        | Trees | 300 |
        | Max Depth | 6 |
        | Learning Rate | 0.05 |
        | Imbalance Handling | scale_pos_weight |
        | Explainability | SHAP TreeExplainer |
        """)

    with mi2:
        st.markdown("**Isolation Forest Anomaly Detector**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | Isolation Forest |
        | Trees | 200 |
        | Contamination | 8% |
        | Use Case | Unsupervised anomaly detection |
        | Output | Anomaly score (-1 to 0) |
        """)

    # Feature importances bar chart
    fi_path = MODELS_DIR / "feature_importances.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)

        feats = [k.replace("_", " ").title() for k in list(fi.keys())[:15]]
        imps  = list(fi.values())[:15]

        fig_fi = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h",
            marker_color="#58a6ff",
            text=[f"{v:.4f}" for v in imps],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title="Global Feature Importance (XGBoost)",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font={"color": "#e6edf3"}, height=450,
            xaxis={"gridcolor": "#21262d", "title": "Importance Score"},
            yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
            margin=dict(l=10, r=80, t=40, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Model metrics table
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        st.markdown("**Performance Metrics**")
        met_df = pd.DataFrame([{
            "Metric": k.replace("_", " ").title(),
            "Value": f"{v:.4f}" if isinstance(v, float) else v
        } for k, v in m.items() if not isinstance(v, dict)])
        st.dataframe(met_df, use_container_width=True, hide_index=True)

    # Pipeline diagram
    st.markdown('<p class="section-header">🔄 System Pipeline</p>', unsafe_allow_html=True)
    st.markdown("""
    ```
    Raw Batch Parameters
          │
          ▼
    Feature Engineering
    (temp_deviation, moisture_risk, process_score, api_deviation, ph_deviation)
          │
          ▼
    StandardScaler (normalization)
          │
          ├──────────────────────────────┐
          ▼                              ▼
    XGBoost Classifier          Isolation Forest
    (Failure Probability)       (Anomaly Detection)
          │                              │
          └──────────────┬───────────────┘
                         ▼
                  SHAP Explainer
                  (Feature Attribution)
                         │
                         ▼
                  Groq LLaMA3-70b
                  (AI Recommendation)
                         │
                         ▼
                  Streamlit Dashboard
    ```
    """)
