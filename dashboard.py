"""
dashboard.py
============
Streamlit visual dashboard for the Crop Recommendation system.

Features:
  • Live sensor gauges  (temperature, humidity, soil moisture)
  • Crop recommendation card with confidence bar chart
  • Historical trend charts from MongoDB history
  • Manual simulator with sliders
  • Auto-refresh toggle

Run (after starting app.py):
    streamlit run dashboard.py
"""

import os
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
# On Streamlit Cloud: set FLASK_API_URL in Secrets to your Render API URL
API_BASE   = os.getenv("FLASK_API_URL", "http://localhost:5000")
PAGE_TITLE = "🌾 Crop Recommendation AI"
REFRESH_S  = 30

# Crop emoji map
CROP_EMOJI = {
    "rice": "🌾", "wheat": "🌿", "maize": "🌽", "cotton": "🌸",
    "sugarcane": "🎋", "jute": "🌱", "coffee": "☕", "coconut": "🥥",
    "papaya": "🍈", "orange": "🍊", "apple": "🍎", "muskmelon": "🍈",
    "watermelon": "🍉", "grapes": "🍇", "mango": "🥭", "banana": "🍌",
    "pomegranate": "🍎", "lentil": "🌿", "blackgram": "🌿",
    "mothbeans": "🌿", "mungbean": "🌿", "pigeonpeas": "🌿",
    "kidneybeans": "🫘", "chickpea": "🫘",
}

# ─── Streamlit page setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #0d2b1a 100%);
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.crop-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 20px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
}

.crop-name {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.crop-emoji { font-size: 4rem; }

.confidence-badge {
    display: inline-block;
    background: rgba(34,197,94,0.2);
    border: 1px solid #4ade80;
    color: #4ade80;
    border-radius: 999px;
    padding: 4px 18px;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 8px;
}

.sensor-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.sensor-value {
    font-size: 2rem;
    font-weight: 700;
    color: #22d3ee;
}

.sensor-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.status-dot {
    width: 10px; height: 10px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_rate = st.slider("Refresh interval (s)", 10, 120, REFRESH_S)
    st.divider()
    st.markdown("## 🧪 Manual Simulator")
    sim_temp     = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
    sim_humidity = st.slider("Humidity (%)",     0.0, 100.0, 70.0, 1.0)
    sim_moisture = st.slider("Soil Moisture (%)",0.0, 100.0, 55.0, 1.0)

    if st.button("🔮 Simulate", use_container_width=True):
        st.session_state["sim_result"] = None
        try:
            resp = requests.post(
                f"{API_BASE}/predict",
                json={"temperature": sim_temp, "humidity": sim_humidity,
                      "soilMoisture": sim_moisture},
                timeout=5,
            )
            st.session_state["sim_result"] = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")

    if st.session_state.get("sim_result"):
        r = st.session_state["sim_result"]
        emoji = CROP_EMOJI.get(r["recommended_crop"], "🌱")
        st.success(f"{emoji}  **{r['recommended_crop'].upper()}** — {r['confidence']}")
        for item in r["top_crops"][:3]:
            st.progress(item["probability"],
                        text=f"{item['crop']:15s} {item['confidence']}")

    st.divider()
    st.markdown("**API Base URL**")
    api_input = st.text_input("", value=API_BASE, key="api_url")
    API_BASE = api_input


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-size:2.4rem;font-weight:800;margin-bottom:4px;'>
    🌾 Crop Recommendation AI
</h1>
<p style='color:rgba(255,255,255,0.45);margin-top:0;'>
    <span class='status-dot'></span>
    Powered by Random Forest · Live Data from ESP32 / MongoDB Atlas
</p>
""", unsafe_allow_html=True)

st.divider()

# ─── Fetch live recommendation ─────────────────────────────────────────────
@st.cache_data(ttl=refresh_rate)
def fetch_recommendation():
    try:
        r = requests.get(f"{API_BASE}/recommend", timeout=8)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Flask API. Make sure `python app.py` is running."
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=refresh_rate)
def fetch_history(n=50):
    try:
        r = requests.get(f"{API_BASE}/history?n={n}", timeout=10)
        return r.json().get("readings", []), None
    except Exception as e:
        return [], str(e)


data, err = fetch_recommendation()
history, hist_err = fetch_history(50)

# ─── Error banner ─────────────────────────────────────────────────────────────
if err:
    st.error(f"⚠️  {err}")
    st.info("💡  Start the Flask API: `python app.py`")
    st.stop()

# ─── Live sensor row ──────────────────────────────────────────────────────────
sensor = data.get("sensor_input", {})
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='sensor-card'>
        <div class='sensor-value'>{sensor.get('temperature', '--')}°C</div>
        <div class='sensor-label'>🌡️ Temperature</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='sensor-card'>
        <div class='sensor-value'>{sensor.get('humidity', '--')}%</div>
        <div class='sensor-label'>💧 Humidity</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='sensor-card'>
        <div class='sensor-value'>{sensor.get('soilMoisture', '--')}%</div>
        <div class='sensor-label'>🌱 Soil Moisture</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='sensor-card'>
        <div class='sensor-value'>{sensor.get('rainfall_equiv', '--'):.0f}mm</div>
        <div class='sensor-label'>🌧️ Rainfall Equiv</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Recommendation card + bar chart ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    crop  = data.get("recommended_crop", "unknown")
    emoji = CROP_EMOJI.get(crop, "🌱")
    conf  = data.get("confidence", "N/A")
    ts    = data.get("data_timestamp", "")

    st.markdown(f"""
    <div class='crop-card'>
        <div class='crop-emoji'>{emoji}</div>
        <p class='crop-name'>{crop.upper()}</p>
        <span class='confidence-badge'>✅ {conf} Confidence</span>
        <p style='color:rgba(255,255,255,0.3);font-size:0.78rem;margin-top:12px;'>
            Last reading: {ts[:19] if ts else 'N/A'}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    top_crops = data.get("top_crops", [])
    if top_crops:
        df_top = pd.DataFrame(top_crops)
        df_top["probability_pct"] = df_top["probability"] * 100

        fig = go.Figure(go.Bar(
            x=df_top["probability_pct"],
            y=df_top["crop"],
            orientation="h",
            marker=dict(
                color=df_top["probability_pct"],
                colorscale=[[0, "#134e4a"], [0.5, "#059669"], [1, "#4ade80"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in df_top["probability_pct"]],
            textposition="outside",
            textfont=dict(color="white", size=13),
        ))
        fig.update_layout(
            title="Top Crop Probabilities",
            title_font=dict(color="white", size=15),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
            margin=dict(l=10, r=40, t=40, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── Gauge charts ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("<div class='section-header'>📊 Sensor Gauges</div>", unsafe_allow_html=True)

gc1, gc2, gc3 = st.columns(3)

def make_gauge(value, title, unit, max_val, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": unit, "font": {"size": 28, "color": "white"}},
        title={"text": title, "font": {"size": 14, "color": "rgba(255,255,255,0.6)"}},
        gauge={
            "axis": {"range": [0, max_val], "tickfont": {"color": "rgba(255,255,255,0.4)"}},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "rgba(255,50,50,0.15)"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(250,200,50,0.10)"},
                {"range": [max_val * 0.66, max_val], "color": "rgba(50,200,100,0.10)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=220,
    )
    return fig

with gc1:
    st.plotly_chart(
        make_gauge(sensor.get("temperature", 0), "Temperature", "°C", 50, "#f97316"),
        use_container_width=True
    )
with gc2:
    st.plotly_chart(
        make_gauge(sensor.get("humidity", 0), "Humidity", "%", 100, "#22d3ee"),
        use_container_width=True
    )
with gc3:
    st.plotly_chart(
        make_gauge(sensor.get("soilMoisture", 0), "Soil Moisture", "%", 100, "#4ade80"),
        use_container_width=True
    )

# ─── Historical charts ────────────────────────────────────────────────────────
if history:
    st.divider()
    st.markdown("<div class='section-header'>📈 Historical Readings (last 50)</div>",
                unsafe_allow_html=True)

    df_hist = pd.DataFrame(history)
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")
    df_hist = df_hist.sort_values("timestamp")

    # Sensor lines chart
    fig_hist = go.Figure()
    for col, color, name in [
        ("temperature",  "#f97316", "Temperature (°C)"),
        ("humidity",     "#22d3ee", "Humidity (%)"),
        ("soilMoisture", "#4ade80", "Soil Moisture (%)"),
    ]:
        fig_hist.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist[col],
            name=name,
            line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=4),
        ))

    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Crop frequency pie
    crop_counts = df_hist["recommended_crop"].value_counts().reset_index()
    crop_counts.columns = ["crop", "count"]

    col_pie, col_table = st.columns([1, 1])
    with col_pie:
        fig_pie = px.pie(
            crop_counts, names="crop", values="count",
            title="Crop Recommendation Distribution",
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font=dict(size=14, color="white"),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.markdown("##### Recent Predictions")
        display_cols = ["timestamp", "temperature", "humidity", "soilMoisture",
                        "recommended_crop", "confidence"]
        available   = [c for c in display_cols if c in df_hist.columns]
        st.dataframe(
            df_hist[available].tail(10).sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=280,
        )

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.cache_data.clear()
    st.rerun()

# Footer
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.06);margin-top:40px;'/>
<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.78rem;'>
    Crop Recommendation AI · ESP32 + MongoDB Atlas + scikit-learn · Built with Streamlit
</p>
""", unsafe_allow_html=True)
