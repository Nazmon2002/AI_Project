import streamlit as st
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense · Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1f0e 0%, #102010 40%, #0a1a10 100%);
    color: #e8f0e9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 28, 12, 0.95) !important;
    border-right: 1px solid rgba(94, 190, 100, 0.2);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #6fcf76;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Number inputs & sliders ── */
input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(94,190,100,0.3) !important;
    border-radius: 8px !important;
    color: #e8f0e9 !important;
}
input[type="number"]:focus {
    border-color: #6fcf76 !important;
    box-shadow: 0 0 0 2px rgba(111,207,118,0.15) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #6fcf76 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3a9e42, #6fcf76) !important;
    color: #0d1f0e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 0 !important;
    margin-top: 1rem;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(111,207,118,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(111,207,118,0.38) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(58,158,66,0.18), rgba(111,207,118,0.10));
    border: 1px solid rgba(111,207,118,0.45);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-top: 2rem;
    text-align: center;
    backdrop-filter: blur(8px);
}
.result-card .label {
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6fcf76;
    margin-bottom: 0.5rem;
}
.result-card .crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 900;
    color: #d4f7d6;
    line-height: 1.1;
    text-transform: capitalize;
}
.result-card .score-line {
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: rgba(200,230,200,0.65);
}

/* ── Hero header ── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #d4f7d6;
    line-height: 1.15;
    margin: 0;
}
.hero .sub {
    font-size: 1rem;
    color: rgba(200,230,200,0.6);
    margin-top: 0.5rem;
}
.badge {
    display: inline-block;
    background: rgba(111,207,118,0.15);
    border: 1px solid rgba(111,207,118,0.35);
    color: #6fcf76;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

/* ── Divider ── */
hr { border-color: rgba(94,190,100,0.15) !important; }

/* ── Info boxes ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
    margin-top: 1.5rem;
}
.info-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(94,190,100,0.18);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.info-box .param { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6fcf76; }
.info-box .val   { font-size: 1.25rem; font-weight: 600; color: #d4f7d6; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── Data & algorithm ──────────────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base():
    df = pd.read_csv("Crop_recommendation.csv")
    return df.groupby("label").mean().reset_index()

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def calculate_distance(user: dict, crop: dict) -> float:
    return (
        abs(user["N"]           - crop["N"])           * 1.0  +
        abs(user["P"]           - crop["P"])           * 1.0  +
        abs(user["K"]           - crop["K"])           * 1.0  +
        abs(user["temperature"] - crop["temperature"]) * 2.0  +
        abs(user["humidity"]    - crop["humidity"])    * 0.5  +
        abs(user["ph"]          - crop["ph"])          * 10.0 +
        abs(user["rainfall"]    - crop["rainfall"])    * 0.3
    )

def iddfs_recommend(user_input: dict, knowledge_base: pd.DataFrame, max_depth: int = 10):
    """
    IDDFS over crops sorted by distance at each depth level.
    At depth d we keep only crops whose distance ≤ best_so_far * (1 + 1/d),
    progressively tightening the search window.
    """
    candidates = knowledge_base.to_dict("records")
    distances  = {row["label"]: calculate_distance(user_input, row) for row in candidates}

    best_crop, best_dist = None, float("inf")

    for depth in range(1, max_depth + 1):
        threshold = best_dist * (1 + 1 / depth) if best_dist < float("inf") else float("inf")
        for crop in candidates:
            d = distances[crop["label"]]
            if d < threshold:
                if d < best_dist:
                    best_dist = d
                    best_crop = crop["label"]
        if best_crop is not None:
            # Once we have a stable best within tightening threshold, converge
            if depth > 3:
                break

    return best_crop, best_dist


# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Soil & Climate Inputs")
    st.markdown("Adjust the parameters to match your field conditions.")
    st.markdown("---")

    N           = st.number_input("Nitrogen (N)",           min_value=0.0,  max_value=200.0, value=90.0,  step=1.0,  format="%.1f")
    P           = st.number_input("Phosphorus (P)",         min_value=0.0,  max_value=200.0, value=42.0,  step=1.0,  format="%.1f")
    K           = st.number_input("Potassium (K)",          min_value=0.0,  max_value=210.0, value=43.0,  step=1.0,  format="%.1f")
    temperature = st.number_input("Temperature (°C)",       min_value=0.0,  max_value=55.0,  value=20.8,  step=0.1,  format="%.1f")
    humidity    = st.number_input("Humidity (%)",           min_value=0.0,  max_value=100.0, value=82.0,  step=0.5,  format="%.1f")
    ph          = st.slider(      "Soil pH",                min_value=3.0,  max_value=10.0,  value=6.5,   step=0.05, format="%.2f")
    rainfall    = st.number_input("Rainfall (mm)",          min_value=0.0,  max_value=300.0, value=202.9, step=1.0,  format="%.1f")

    st.markdown("---")
    predict_btn = st.button("🌾 Predict Best Crop")


# ── Main panel ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">IDDFS · AI Recommendation Engine</div>
  <h1>CropSense</h1>
  <p class="sub">Enter your soil & climate data on the left — we'll find the ideal crop for your land.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    st.markdown("**Algorithm**")
    st.markdown("Iterative Deepening DFS with weighted Euclidean distance across 7 agro-climatic features.")
with col2:
    st.markdown("**Dataset**")
    st.markdown("`Crop_recommendation.csv` — 22 crop classes, grouped by mean feature values.")
with col3:
    st.markdown("**Features**")
    st.markdown("N · P · K · Temperature · Humidity · pH · Rainfall")

st.markdown("---")

# ── Run prediction ────────────────────────────────────────────────────────────
if predict_btn:
    try:
        kb = load_knowledge_base()
    except FileNotFoundError:
        st.error("⚠️ `Crop_recommendation.csv` not found. Make sure it's in the same folder as `app.py`.")
        st.stop()

    user_input = {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    with st.spinner("Running IDDFS search across crop profiles…"):
        crop, dist = iddfs_recommend(user_input, kb)

    emoji_map = {
        "rice":"🌾","maize":"🌽","chickpea":"🫘","kidneybeans":"🫘","pigeonpeas":"🫘",
        "mothbeans":"🫘","mungbean":"🫘","blackgram":"🫘","lentil":"🫘","pomegranate":"🍎",
        "banana":"🍌","mango":"🥭","grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
        "apple":"🍎","orange":"🍊","papaya":"🍈","coconut":"🥥","cotton":"🌿",
        "jute":"🌿","coffee":"☕",
    }
    icon = emoji_map.get(crop.lower(), "🌱") if crop else "🌱"

    if crop:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Recommended Crop</div>
            <div class="crop-name">{icon} {crop.title()}</div>
            <div class="score-line">Weighted distance score: <strong>{dist:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Show input summary
        st.markdown("#### Your Input Parameters")
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-box"><div class="param">Nitrogen</div><div class="val">{N:.1f}</div></div>
            <div class="info-box"><div class="param">Phosphorus</div><div class="val">{P:.1f}</div></div>
            <div class="info-box"><div class="param">Potassium</div><div class="val">{K:.1f}</div></div>
            <div class="info-box"><div class="param">Temp °C</div><div class="val">{temperature:.1f}</div></div>
            <div class="info-box"><div class="param">Humidity %</div><div class="val">{humidity:.1f}</div></div>
            <div class="info-box"><div class="param">Soil pH</div><div class="val">{ph:.2f}</div></div>
            <div class="info-box"><div class="param">Rainfall mm</div><div class="val">{rainfall:.1f}</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Could not determine a recommendation. Please check your inputs.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color: rgba(200,230,200,0.35);">
        <div style="font-size:4rem;">🌱</div>
        <div style="font-size:1rem; margin-top:1rem; letter-spacing:0.05em;">
            Set your soil & climate parameters in the sidebar, then click <strong style="color:rgba(111,207,118,0.6)">Predict Best Crop</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
