import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO

# --------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiac Risk Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
def load_css(file_path: str):
    css_file = Path(file_path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------------------------------
# Feature order the model/scaler were trained on — never change this order
# --------------------------------------------------------------------------
FEATURE_ORDER = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium",
]

FEATURE_LABELS = {
    "Age": "Age",
    "Sex": "Sex",
    "Chest pain type": "Chest pain type",
    "BP": "Resting blood pressure",
    "Cholesterol": "Cholesterol",
    "FBS over 120": "Fasting blood sugar > 120",
    "EKG results": "Resting EKG",
    "Max HR": "Max heart rate",
    "Exercise angina": "Exercise-induced angina",
    "ST depression": "ST depression",
    "Slope of ST": "Slope of peak ST segment",
    "Number of vessels fluro": "Major vessels (fluoroscopy)",
    "Thallium": "Thallium stress result",
}

# --------------------------------------------------------------------------
# Load model artifacts — try local files first, fall back to manual upload
# so the app keeps working even if the .pkl files aren't co-located.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_local_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("heart_disease_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return scaler, model, label_encoder

def try_load():
    try:
        return load_local_artifacts(), None
    except Exception as e:
        return None, str(e)

artifacts, load_error = try_load()

if artifacts is None:
    st.sidebar.warning("Model files weren't found next to app.py.")
    st.sidebar.markdown("**Upload them here instead:**")
    up_scaler = st.sidebar.file_uploader("scaler.pkl", type="pkl", key="up_scaler")
    up_model = st.sidebar.file_uploader("heart_disease_model.pkl", type="pkl", key="up_model")
    up_le = st.sidebar.file_uploader("label_encoder.pkl", type="pkl", key="up_le")
    if up_scaler and up_model and up_le:
        try:
            scaler = joblib.load(BytesIO(up_scaler.read()))
            model = joblib.load(BytesIO(up_model.read()))
            label_encoder = joblib.load(BytesIO(up_le.read()))
            artifacts = (scaler, model, label_encoder)
        except Exception as e:
            st.sidebar.error(f"Couldn't load uploaded files: {e}")

if artifacts is None:
    st.error(
        "⚠️ Model artifacts aren't loaded yet. Place `scaler.pkl`, "
        "`heart_disease_model.pkl`, and `label_encoder.pkl` in the same "
        "folder as `app.py`, or upload them in the sidebar."
    )
    st.stop()

scaler, model, label_encoder = artifacts

# --------------------------------------------------------------------------
# Encoding maps (friendly UI labels -> the numeric codes the model expects)
# --------------------------------------------------------------------------
SEX_MAP = {"Male": 1, "Female": 0}
CHEST_PAIN_MAP = {
    "Typical angina": 1, "Atypical angina": 2,
    "Non-anginal pain": 3, "Asymptomatic": 4,
}
FBS_MAP = {"No": 0, "Yes": 1}
EKG_MAP = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
ANGINA_MAP = {"No": 0, "Yes": 1}
SLOPE_MAP = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THALLIUM_MAP = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

# --------------------------------------------------------------------------
# Header — animated ECG waveform
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="monitor-header">
        <div class="monitor-header-top">
            <div class="monitor-title-block">
                <span class="monitor-eyebrow">CLINICAL DECISION SUPPORT</span>
                <h1>Cardiac Risk Monitor</h1>
                <p>Statistical estimate of heart disease presence from 13 clinical measurements.</p>
            </div>
            <div class="monitor-status">
                <span class="dot"></span> MODEL ONLINE
            </div>
        </div>
        <div class="ekg-wrap">
            <svg class="ekg-line" viewBox="0 0 1200 90" preserveAspectRatio="none">
                <path d="M0,45 L140,45 L165,45 L180,20 L200,75 L220,10 L240,45 L260,45 L400,45
                         L540,45 L565,45 L580,20 L600,75 L620,10 L640,45 L660,45 L800,45
                         L940,45 L965,45 L980,20 L1000,75 L1020,10 L1040,45 L1060,45 L1200,45" />
            </svg>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫀 About")
    st.markdown(
        "A **Logistic Regression** model trained on the Statlog Heart "
        "Disease dataset estimates the probability of heart disease "
        "presence from 13 clinical features."
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown(
        "Educational tool only — **not** a medical diagnosis. "
        "Always consult a qualified healthcare provider."
    )
    st.markdown("---")
    st.markdown("### 🔧 Model")
    c1, c2 = st.columns(2)
    c1.metric("Algorithm", "LogReg")
    c2.metric("Features", len(FEATURE_ORDER))
    st.caption(f"Classes: {', '.join(model.classes_)}")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# --------------------------------------------------------------------------
# Input layout: form (left) + live vitals summary (right)
# --------------------------------------------------------------------------
left, right = st.columns([2.1, 1], gap="large")

with left:
    st.markdown('<div class="panel-label">01 · Demographics</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=1, max_value=120, value=54, step=1)
        sex = c2.selectbox("Sex", options=["Male", "Female"])

    st.markdown('<div class="panel-label">02 · Vitals</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        bp = c1.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=130, step=1)
        cholesterol = c2.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=246, step=1)
        max_hr = c3.number_input("Max heart rate", min_value=50, max_value=250, value=150, step=1)
        c4, c5 = st.columns(2)
        fbs = c4.selectbox("Fasting blood sugar > 120?", options=["No", "Yes"])
        ekg = c5.selectbox("Resting EKG", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

    st.markdown('<div class="panel-label">03 · Cardiac testing</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        chest_pain = c1.selectbox(
            "Chest pain type",
            options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
        )
        exercise_angina = c2.selectbox("Exercise-induced angina?", options=["No", "Yes"])
        c3, c4 = st.columns(2)
        st_depression = c3.number_input(
            "ST depression (exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f"
        )
        slope = c4.selectbox("Slope of peak ST segment", options=["Upsloping", "Flat", "Downsloping"])
        c5, c6 = st.columns(2)
        vessels = c5.selectbox("Major vessels (fluoroscopy)", options=[0, 1, 2, 3])
        thallium = c6.selectbox("Thallium stress result", options=["Normal", "Fixed defect", "Reversible defect"])

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
    bcol1, bcol2 = st.columns([3, 1])
    predict_clicked = bcol1.button("🔍 Run risk assessment", use_container_width=True, type="primary")
    reset_clicked = bcol2.button("↺ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.result = None
        st.rerun()

# --------------------------------------------------------------------------
# Build input row (used for both the live summary and the prediction)
# --------------------------------------------------------------------------
input_dict = {
    "Age": age, "Sex": SEX_MAP[sex], "Chest pain type": CHEST_PAIN_MAP[chest_pain],
    "BP": bp, "Cholesterol": cholesterol, "FBS over 120": FBS_MAP[fbs],
    "EKG results": EKG_MAP[ekg], "Max HR": max_hr, "Exercise angina": ANGINA_MAP[exercise_angina],
    "ST depression": st_depression, "Slope of ST": SLOPE_MAP[slope],
    "Number of vessels fluro": vessels, "Thallium": THALLIUM_MAP[thallium],
}
input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

# --------------------------------------------------------------------------
# Right column: live vitals chip summary
# --------------------------------------------------------------------------
with right:
    st.markdown('<div class="panel-label">LIVE SUMMARY</div>', unsafe_allow_html=True)
    chips = f"""
    <div class="chip-panel">
        <div class="chip"><span>Age</span><strong>{age}</strong></div>
        <div class="chip"><span>Sex</span><strong>{sex}</strong></div>
        <div class="chip"><span>BP</span><strong>{bp} mmHg</strong></div>
        <div class="chip"><span>Cholesterol</span><strong>{cholesterol} mg/dl</strong></div>
        <div class="chip"><span>Max HR</span><strong>{max_hr} bpm</strong></div>
        <div class="chip"><span>Chest pain</span><strong>{chest_pain}</strong></div>
        <div class="chip"><span>Ex. angina</span><strong>{exercise_angina}</strong></div>
        <div class="chip"><span>ST depression</span><strong>{st_depression}</strong></div>
        <div class="chip"><span>Thallium</span><strong>{thallium}</strong></div>
    </div>
    """
    st.markdown(chips, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Run prediction
# --------------------------------------------------------------------------
if predict_clicked:
    try:
        scaled_input = scaler.transform(input_df)

        # model.classes_ are already the human-readable labels
        prediction_label = model.predict(scaled_input)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(scaled_input)[0]
            proba = dict(zip(model.classes_, p))

        # Per-feature contribution to the "Presence" log-odds, for interpretability
        contributions = None
        if hasattr(model, "coef_"):
            presence_idx = list(model.classes_).index("Presence") if "Presence" in model.classes_ else -1
            coefs = model.coef_[0] if presence_idx != 0 else -model.coef_[0]
            contrib_values = coefs * scaled_input[0]
            contributions = sorted(
                zip(FEATURE_ORDER, contrib_values), key=lambda x: abs(x[1]), reverse=True
            )

        st.session_state.result = {
            "label": prediction_label,
            "proba": proba,
            "contributions": contributions,
            "input_df": input_df,
        }
    except Exception as e:
        st.session_state.result = None
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.result

if result:
    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">04 · Readout</div>', unsafe_allow_html=True)

    is_positive = result["label"].strip().lower() == "presence"
    presence_prob = result["proba"].get("Presence", 0.0) if result["proba"] else None
    pct = presence_prob * 100 if presence_prob is not None else None

    gcol, vcol = st.columns([1.1, 1], gap="large")

    with gcol:
        if pct is not None:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 44, "family": "IBM Plex Mono, monospace", "color": "#0B1B2B"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5B6B7A"},
                    "bar": {"color": "#0B1B2B", "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "#DCF6E6"},
                        {"range": [40, 70], "color": "#FDECC8"},
                        {"range": [70, 100], "color": "#FBDADC"},
                    ],
                    "threshold": {
                        "line": {"color": "#E5484D", "width": 3},
                        "thickness": 0.85,
                        "value": pct,
                    },
                },
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter, sans-serif"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Estimated probability of heart disease **presence**")

    with vcol:
        if is_positive:
            st.markdown(
                f"""<div class="verdict verdict-positive">
                        <div class="verdict-icon">⚠️</div>
                        <div>
                            <div class="verdict-title">Presence indicated</div>
                            <div class="verdict-sub">Model class: <strong>{result['label']}</strong></div>
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="verdict verdict-negative">
                        <div class="verdict-icon">✅</div>
                        <div>
                            <div class="verdict-title">Absence indicated</div>
                            <div class="verdict-sub">Model class: <strong>{result['label']}</strong></div>
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )

        if result["contributions"]:
            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
            st.markdown("**Top contributing factors**")
            max_abs = max(abs(v) for _, v in result["contributions"][:5]) or 1.0
            for feat, val in result["contributions"][:5]:
                width = min(abs(val) / max_abs * 100, 100)
                direction = "up" if val > 0 else "down"
                bar_class = "bar-up" if direction == "up" else "bar-down"
                arrow = "▲" if direction == "up" else "▼"
                st.markdown(
                    f"""
                    <div class="factor-row">
                        <div class="factor-label">{FEATURE_LABELS.get(feat, feat)}</div>
                        <div class="factor-track">
                            <div class="factor-fill {bar_class}" style="width:{width:.0f}%"></div>
                        </div>
                        <div class="factor-arrow {bar_class}">{arrow}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("▲ pushes risk up · ▼ pushes risk down, relative to this patient's inputs")

    with st.expander("Full class probabilities & raw model input"):
        if result["proba"]:
            proba_df = pd.DataFrame(
                {"Class": list(result["proba"].keys()),
                 "Probability": [f"{v*100:.2f}%" for v in result["proba"].values()]}
            )
            st.table(proba_df)
        st.dataframe(result["input_df"], use_container_width=True)

    st.caption(
        "⚠️ This output is generated by a statistical model, not a clinician. "
        "It is not a diagnosis — please consult a qualified healthcare professional."
    )
else:
    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    st.info("Fill in the patient's details and click **Run risk assessment** to see the readout.")