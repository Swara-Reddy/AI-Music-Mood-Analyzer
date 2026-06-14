import streamlit as st
import joblib
import tempfile
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────── PAGE CONFIG ─────────────────────────── #

st.set_page_config(
    page_title="MoodWave · AI Music Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── BASE STYLES ─────────────────────────── #

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&family=Inter:wght@300;400;500;600;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

/* ── Root dark canvas ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #1e1e2e !important;
}

[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Main content wrapper ── */
.block-container {
    padding: 2rem 3rem 4rem 3rem !important;
    max-width: 1100px !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
    color: #ffffff !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #16161f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #13131c !important;
    border: 2px dashed #2a2a3e !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6c47ff !important;
}

/* ── Audio player ── */
audio {
    width: 100%;
    border-radius: 50px;
    filter: invert(1) hue-rotate(180deg);
}

/* ── Buttons ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #6c47ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    opacity: 0.85 !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    border-radius: 50px;
    height: 6px;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #1e1e2e !important;
    margin: 1.5rem 0 !important;
}

/* ── Success / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: none !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #13131c !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 14px !important;
}

/* ── Bar chart ── */
[data-testid="stArrowVegaLiteChart"] canvas {
    border-radius: 12px;
}

/* ── Sidebar logo area ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #2a2a3e;
    margin-bottom: 1.5rem;
}
.sidebar-brand-text {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #6c47ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #1a1a26;
    font-size: 0.85rem;
}
.sidebar-stat-label { color: #666; }
.sidebar-stat-value {
    font-weight: 600;
    color: #ccc;
}

/* ── Section headers ── */
.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6c47ff;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.025em;
    margin-bottom: 1.2rem;
}

/* ── Mood badge ── */
.mood-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 0.75rem 1.5rem;
    border-radius: 50px;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}

/* ── Song card ── */
.song-card {
    background: #13131c;
    border: 1px solid #2a2a3e;
    border-radius: 18px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: background 0.2s, border-color 0.2s;
}
.song-card:hover {
    background: #1a1a28;
    border-color: #3a3a58;
}
.song-rank {
    font-size: 1.1rem;
    font-weight: 800;
    width: 28px;
    text-align: center;
    opacity: 0.35;
}
.song-info { flex: 1; }
.song-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #fff;
    margin-bottom: 2px;
}
.song-meta {
    font-size: 0.78rem;
    color: #666;
}
.song-sim {
    font-size: 0.8rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    background: #1e1e2e;
    color: #aaa;
}

/* ── AI insights ── */
.insight-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.25rem;
    border: 1px solid #2a2a3e;
    background: #13131c;
    color: #ccc;
}

/* ── Hero upload area ── */
.hero-section {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: #666;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Matplotlib figure dark theme ── */
.stPlotlyChart, .stImage { border-radius: 16px; overflow: hidden; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ─────────────────────────── MOOD PALETTE ─────────────────────────── #

MOOD_THEMES = {
    "happy": {
        "gradient": "linear-gradient(135deg, #f59e0b, #fbbf24, #fde68a)",
        "accent": "#f59e0b",
        "glow": "rgba(245,158,11,0.25)",
        "badge_bg": "rgba(245,158,11,0.15)",
        "badge_text": "#fde68a",
        "icon": "😊",
        "label": "Happy",
    },
    "sad": {
        "gradient": "linear-gradient(135deg, #1d4ed8, #3b82f6, #93c5fd)",
        "accent": "#3b82f6",
        "glow": "rgba(59,130,246,0.25)",
        "badge_bg": "rgba(59,130,246,0.15)",
        "badge_text": "#93c5fd",
        "icon": "😢",
        "label": "Sad",
    },
    "romantic": {
        "gradient": "linear-gradient(135deg, #be185d, #ec4899, #fbcfe8)",
        "accent": "#ec4899",
        "glow": "rgba(236,72,153,0.25)",
        "badge_bg": "rgba(236,72,153,0.15)",
        "badge_text": "#fbcfe8",
        "icon": "❤️",
        "label": "Romantic",
    },
    "aggressive": {
        "gradient": "linear-gradient(135deg, #991b1b, #ef4444, #fca5a5)",
        "accent": "#ef4444",
        "glow": "rgba(239,68,68,0.25)",
        "badge_bg": "rgba(239,68,68,0.15)",
        "badge_text": "#fca5a5",
        "icon": "🔥",
        "label": "Aggressive",
    },
    "dramatic": {
        "gradient": "linear-gradient(135deg, #4c1d95, #8b5cf6, #ddd6fe)",
        "accent": "#8b5cf6",
        "glow": "rgba(139,92,246,0.25)",
        "badge_bg": "rgba(139,92,246,0.15)",
        "badge_text": "#ddd6fe",
        "icon": "🎭",
        "label": "Dramatic",
    },
}

def set_mood_theme(mood):
    t = MOOD_THEMES.get(mood, MOOD_THEMES["dramatic"])
    st.markdown(f"""
    <style>
    /* Dynamic accent glow on upload zone */
    [data-testid="stFileUploader"] {{
        box-shadow: 0 0 40px {t['glow']} !important;
        border-color: {t['accent']}44 !important;
    }}
    /* Mood-colored progress bar fill */
    [data-testid="stProgress"] > div > div > div {{
        background: {t['accent']} !important;
    }}
    /* Accent on sidebar stat values */
    .sidebar-stat-value {{ color: {t['accent']} !important; }}
    /* Metric accent border */
    [data-testid="metric-container"] {{
        border-color: {t['accent']}33 !important;
        box-shadow: 0 0 20px {t['glow']};
    }}
    </style>
    """, unsafe_allow_html=True)
    return t

# ─────────────────────────── LOAD FILES ─────────────────────────── #

model = joblib.load("notebooks/music_mood_model.pkl")
scaler = joblib.load("notebooks/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

song_database = pd.read_csv("data/music_features_dataset.csv")

# ─────────────────────────── SIDEBAR ─────────────────────────── #

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size:1.6rem">🎵</span>
        <span class="sidebar-brand-text">MoodWave</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Model</span>
        <span class="sidebar-stat-value">Random Forest</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Accuracy</span>
        <span class="sidebar-stat-value">90%</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Songs</span>
        <span class="sidebar-stat-value">500</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Features</span>
        <span class="sidebar-stat-value">30</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#444;margin-bottom:0.8rem;">
        FEATURES USED
    </div>
    """, unsafe_allow_html=True)

    feature_tags = ["MFCC", "Chroma", "Tempo", "RMS Energy",
                    "Spectral Centroid", "Bandwidth", "ZCR"]
    tags_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;margin:3px;background:#1a1a26;border:1px solid #2a2a3e;border-radius:20px;font-size:0.75rem;color:#888;">{t}</span>'
        for t in feature_tags
    )
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:#333;text-align:center;line-height:1.6;">
        Built with librosa · sklearn<br>
        <span style="color:#6c47ff">MoodWave v1.0</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── FEATURE EXTRACTION ─────────────────────────── #

def extract_features(file_path):
    audio, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.asarray(tempo).squeeze())
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    features = {
        "tempo": tempo,
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zero_crossing_rate,
        "rms": rms,
        "spectral_bandwidth": spectral_bandwidth,
    }
    for i in range(12):
        features[f"chroma_{i+1}"] = chroma[i]
    for i in range(13):
        features[f"mfcc_{i+1}"] = mfcc_mean[i]
    return features

# ─────────────────────────── PREDICTION ─────────────────────────── #

def predict_mood(file_path):
    features = extract_features(file_path)
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)
    scaled_features = scaler.transform(feature_df)
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    return prediction[0], probabilities[0], features

# ─────────────────────────── RECOMMENDATION ─────────────────────────── #

def recommend_songs(features, predicted_mood, top_n=5):
    feature_data = song_database[feature_columns]
    query = pd.DataFrame([features])
    query = query.reindex(columns=feature_columns, fill_value=0)
    similarities = cosine_similarity(query, feature_data)[0]
    top_indices = similarities.argsort()[::-1]
    recommendations = song_database.iloc[top_indices].copy()
    recommendations["similarity"] = similarities[top_indices]
    recommendations = recommendations[
        recommendations["mood"] == predicted_mood
    ].head(top_n)
    return recommendations

# ─────────────────────────── WAVEFORM ─────────────────────────── #

def plot_waveform(file_path, accent="#6c47ff"):
    audio, sr = librosa.load(file_path)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0d0d14")
    ax.set_facecolor("#0d0d14")
    ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio,
            color=accent, linewidth=0.6, alpha=0.9)
    ax.fill_between(np.linspace(0, len(audio) / sr, len(audio)),
                    audio, alpha=0.15, color=accent)
    ax.set_title("Waveform", color="#666", fontsize=10, pad=8)
    ax.tick_params(colors="#444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")
    ax.set_xlabel("Time (s)", color="#444", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────── SPECTROGRAM ─────────────────────────── #

def plot_spectrogram(file_path):
    audio, sr = librosa.load(file_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0d0d14")
    ax.set_facecolor("#0d0d14")
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log",
                                   ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB",
                 label="dB").ax.yaxis.label.set_color("#666")
    ax.set_title("Spectrogram", color="#666", fontsize=10, pad=8)
    ax.tick_params(colors="#444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")
    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────── HERO + UPLOAD ─────────────────────────── #

st.markdown("""
<div class="hero-section">
    <div class="hero-title">AI Music Analyzer</div>
    <div class="hero-subtitle">Drop a track. Discover its mood. Find your next obsession.</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an audio file — WAV, MP3, FLAC, M4A",
    type=["wav", "mp3", "flac", "m4a", "MPEG"],
    label_visibility="visible",
)

# ─────────────────────────── ANALYSIS ─────────────────────────── #

if uploaded_file is not None:

    st.audio(uploaded_file)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    with st.spinner("Analyzing audio features…"):
        mood, probs, features = predict_mood(temp_path)

    theme = set_mood_theme(mood)
    accent = theme["accent"]

    # ── Mood result banner ── #
    st.markdown(f"""
    <div style="background: {theme['badge_bg']};
                border: 1px solid {accent}44;
                border-radius: 24px;
                padding: 1.8rem 2rem;
                margin: 1.5rem 0;
                display: flex;
                align-items: center;
                gap: 1.2rem;
                box-shadow: 0 0 60px {theme['glow']};">
        <div style="font-size:3.5rem;line-height:1">{theme['icon']}</div>
        <div>
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:4px;">DETECTED MOOD</div>
            <div style="font-size:2.5rem;font-weight:900;letter-spacing:-0.04em;color:{theme['badge_text']};line-height:1">{theme['label']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary / Secondary ── #
    sorted_probs = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="background:#13131c;border:1px solid #2a2a3e;border-radius:16px;padding:1rem 1.4rem;">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#555;margin-bottom:6px;">PRIMARY MOOD</div>
            <div style="font-size:1.25rem;font-weight:800;color:{accent}">{sorted_probs[0][0].title()} &nbsp;<span style="color:#555;font-size:0.9rem;font-weight:500">{sorted_probs[0][1]*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="background:#13131c;border:1px solid #2a2a3e;border-radius:16px;padding:1rem 1.4rem;">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#555;margin-bottom:6px;">SECONDARY MOOD</div>
            <div style="font-size:1.25rem;font-weight:800;color:#888">{sorted_probs[1][0].title()} &nbsp;<span style="color:#555;font-size:0.9rem;font-weight:500">{sorted_probs[1][1]*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Confidence scores ── #
    st.markdown(f"""
    <div class="section-eyebrow">Confidence</div>
    <div class="section-title">Mood Breakdown</div>
    """, unsafe_allow_html=True)

    for mood_name, prob in zip(model.classes_, probs):
        m_theme = MOOD_THEMES.get(mood_name, MOOD_THEMES["dramatic"])
        bar_color = m_theme["accent"]
        pct = f"{prob*100:.1f}%"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.7rem;">
            <span style="width:80px;font-size:0.82rem;color:#888;font-weight:500">{mood_name.title()}</span>
            <div style="flex:1;background:#1a1a26;border-radius:50px;height:6px;overflow:hidden;">
                <div style="width:{prob*100:.1f}%;background:{bar_color};height:100%;border-radius:50px;transition:width 0.4s ease;"></div>
            </div>
            <span style="width:38px;text-align:right;font-size:0.82rem;font-weight:700;color:{bar_color}">{pct}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Bar chart ── #
    confidence_df = pd.DataFrame({"Mood": model.classes_, "Confidence": probs})
    st.bar_chart(confidence_df.set_index("Mood"), color=accent, height=180)

    st.divider()

    # ── Stats ── #
    st.markdown(f"""
    <div class="section-eyebrow">Analysis</div>
    <div class="section-title">Audio Fingerprint</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🥁 Tempo", f"{features['tempo']:.1f} BPM")
    with col2:
        st.metric("⚡ Energy (RMS)", f"{features['rms']:.3f}")
    with col3:
        st.metric("✨ Brightness", f"{features['spectral_centroid']:.0f} Hz")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── AI insights ── #
    st.markdown(f"""<div class="section-eyebrow" style="margin-top:1rem">Insights</div>""", unsafe_allow_html=True)
    insights = []
    if features["tempo"] > 120:
        insights.append("⚡ Fast tempo")
    if features["tempo"] < 90:
        insights.append("🎼 Relaxed tempo")
    if features["rms"] > 0.2:
        insights.append("🔊 High energy")
    if features["spectral_centroid"] > 2000:
        insights.append("✨ Bright sounding")
    if not insights:
        insights.append("🎵 Balanced profile")
    pills = "".join(f'<span class="insight-pill" style="border-color:{accent}33;color:{accent}">{i}</span>' for i in insights)
    st.markdown(f'<div style="margin-bottom:1rem">{pills}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Waveform ── #
    st.markdown(f"""
    <div class="section-eyebrow">Visuals</div>
    <div class="section-title">Waveform</div>
    """, unsafe_allow_html=True)
    plot_waveform(temp_path, accent=accent)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Spectrogram ── #
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.6rem;">Spectrogram</div>', unsafe_allow_html=True)
    plot_spectrogram(temp_path)

    st.divider()

    # ── Download ── #
    report = f"""MoodWave Analysis Report
========================
Mood:        {mood.title()}
Tempo:       {features['tempo']:.2f} BPM
Energy:      {features['rms']:.3f}
Brightness:  {features['spectral_centroid']:.2f} Hz
Primary:     {sorted_probs[0][0].title()} ({sorted_probs[0][1]*100:.1f}%)
Secondary:   {sorted_probs[1][0].title()} ({sorted_probs[1][1]*100:.1f}%)
"""
    st.download_button("📄 Download Report", report, file_name="moodwave_report.txt")

    st.divider()

    # ── Recommendations ── #
    st.markdown(f"""
    <div class="section-eyebrow">Discover</div>
    <div class="section-title">Recommended Tracks</div>
    """, unsafe_allow_html=True)

    recommendations = recommend_songs(features, mood)

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
        sim_pct = int(row["similarity"] * 100)
        filename_clean = row["filename"].replace("_", " ").replace(".wav", "").replace(".mp3", "")
        st.markdown(f"""
        <div class="song-card">
            <div class="song-rank">{i}</div>
            <div style="width:40px;height:40px;border-radius:10px;background:{theme['badge_bg']};
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem;
                        border:1px solid {accent}33;">
                🎵
            </div>
            <div class="song-info">
                <div class="song-title">{filename_clean}</div>
                <div class="song-meta">{row['mood'].title()} · {sim_pct}% match</div>
            </div>
            <div class="song-sim" style="color:{accent};background:{theme['badge_bg']};border:1px solid {accent}33;">
                {sim_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.expander("ℹ️ About MoodWave"):
        st.markdown(f"""
        <div style="color:#aaa;font-size:0.9rem;line-height:1.8">
            <b style="color:#fff">MoodWave</b> uses a Random Forest classifier trained on 500 songs across 5 mood categories.<br><br>
            <b style="color:{accent}">Audio Features:</b> MFCC (13 coefficients), Chroma (12 bins), Tempo, RMS Energy, Spectral Centroid, Spectral Bandwidth, Zero-Crossing Rate<br><br>
            <b style="color:{accent}">Pipeline:</b> librosa extraction → StandardScaler → Random Forest → cosine similarity recommendation
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty state ── #
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;color:#2a2a3e">
        <div style="font-size:4rem;margin-bottom:1rem">🎵</div>
        <div style="font-size:1rem;color:#444">Upload a track above to begin your analysis</div>
    </div>
    """, unsafe_allow_html=True)