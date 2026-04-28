"""
Week 7: Diabetes Risk Early Warning Application
Project: Identifying Early Warning Signals of Diabetes Risk Using Routine Clinical Indicators

Run with:
    streamlit run app.pyÍ

Requirements:
    pip install streamlit scikit-learn pandas numpy matplotlib joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #f7f8fc;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1f3d;
    }
    [data-testid="stSidebar"] * {
        color: #e8edf5 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #a8b8d0 !important;
        font-size: 0.82rem !important;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Risk card styles */
    .risk-card {
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-low    { background: linear-gradient(135deg, #e8f8f0, #c8f0da); border: 2px solid #34c97b; }
    .risk-medium { background: linear-gradient(135deg, #fff8e6, #ffeeb0); border: 2px solid #f5a623; }
    .risk-high   { background: linear-gradient(135deg, #fef0f0, #fdd0d0); border: 2px solid #e84c4c; }

    .risk-label {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        font-weight: 400;
        margin: 0.2rem 0;
    }
    .risk-prob {
        font-size: 3.8rem;
        font-weight: 600;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .risk-low    .risk-prob { color: #1a9e58; }
    .risk-medium .risk-prob { color: #b87800; }
    .risk-high   .risk-prob { color: #c0222a; }
    .risk-low    .risk-label { color: #1a6640; }
    .risk-medium .risk-label { color: #7a4d00; }
    .risk-high   .risk-label { color: #8a1010; }

    .risk-subtitle {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.5rem;
    }

    /* Metric tile */
    .metric-tile {
        background: white;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-name  { font-size: 0.88rem; color: #555; font-weight: 500; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #0f1f3d; }
    .metric-flag  { font-size: 0.75rem; padding: 2px 8px; border-radius: 20px; font-weight: 600; }
    .flag-high    { background: #fde8e8; color: #c0222a; }
    .flag-normal  { background: #e8f4fd; color: #1a6696; }
    .flag-low     { background: #e8f8f0; color: #1a6640; }

    /* Section headers */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #0f1f3d;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e0e4ef;
    }

    /* Info box */
    .info-box {
        background: #eef2fb;
        border-left: 4px solid #3a5fcd;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.9rem;
        color: #2a3a6a;
        margin: 0.8rem 0;
    }

    /* Warning box */
    .warning-box {
        background: #fff7e6;
        border-left: 4px solid #f5a623;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.9rem;
        color: #7a4d00;
        margin: 0.8rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Clinical reference ranges & metadata
# ─────────────────────────────────────────────
FEATURE_META = {
    'Pregnancies':              {'label': 'Number of Pregnancies',         'unit': '',         'min': 0,    'max': 17,   'step': 1,    'default': 3,    'warn_high': 10},
    'Glucose':                  {'label': 'Plasma Glucose',                'unit': 'mg/dL',   'min': 44,   'max': 199,  'step': 1,    'default': 117,  'warn_high': 125,  'warn_low': 70},
    'BloodPressure':            {'label': 'Diastolic Blood Pressure',      'unit': 'mm Hg',   'min': 24,   'max': 122,  'step': 1,    'default': 72,   'warn_high': 90},
    'SkinThickness':            {'label': 'Tricep Skin Fold Thickness',    'unit': 'mm',      'min': 7,    'max': 99,   'step': 1,    'default': 29,   'warn_high': 40},
    'Insulin':                  {'label': '2-Hour Serum Insulin',          'unit': 'μU/mL',   'min': 14,   'max': 846,  'step': 1,    'default': 125,  'warn_high': 166},
    'BMI':                      {'label': 'Body Mass Index',               'unit': 'kg/m²',   'min': 18.2, 'max': 67.1, 'step': 0.1,  'default': 32.3, 'warn_high': 30.0},
    'DiabetesPedigreeFunction': {'label': 'Diabetes Pedigree Function',    'unit': '',        'min': 0.08, 'max': 2.42, 'step': 0.01, 'default': 0.37, 'warn_high': 0.8},
    'Age':                      {'label': 'Age',                           'unit': 'years',   'min': 21,   'max': 81,   'step': 1,    'default': 29,   'warn_high': 45},
}

FEATURE_DESCRIPTIONS = {
    'Pregnancies':              'More pregnancies can indicate gestational diabetes risk history.',
    'Glucose':                  'Fasting plasma glucose ≥ 126 mg/dL is a primary diagnostic criterion for diabetes.',
    'BloodPressure':            'Elevated diastolic BP (≥ 90) is associated with insulin resistance.',
    'SkinThickness':            'Tricep skinfold thickness is a proxy for body fat percentage.',
    'Insulin':                  'Elevated insulin levels may indicate insulin resistance.',
    'BMI':                      'BMI ≥ 30 (obesity) significantly increases Type 2 diabetes risk.',
    'DiabetesPedigreeFunction': 'Encodes genetic predisposition based on family history of diabetes.',
    'Age':                      'Risk of Type 2 diabetes increases significantly after age 45.',
}

# ─────────────────────────────────────────────
# Model loader — trains if no .pkl found
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = './diabetes_model.pkl'
    medians_path = './feature_medians.pkl'

    if os.path.exists(model_path) and os.path.exists(medians_path):
        model   = joblib.load(model_path)
        medians = joblib.load(medians_path)
        return model, medians

    # Fallback: train from scratch if pkl files not present
    if not os.path.exists('diabetes.csv'):
        st.error("diabetes.csv not found. Place it in the same folder as app.py.")
        st.stop()

    df = pd.read_csv('diabetes.csv')
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero] = df[cols_zero].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)

    medians = df[X.columns].median()

    joblib.dump(model, model_path)
    joblib.dump(medians, medians_path)

    return model, medians


@st.cache_data
def load_dataset():
    if not os.path.exists('diabetes.csv'):
        return None
    df = pd.read_csv('diabetes.csv')
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero] = df[cols_zero].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def get_risk_level(prob):
    if prob < 0.30:
        return 'Low', '#34c97b', 'risk-low'
    elif prob < 0.60:
        return 'Moderate', '#f5a623', 'risk-medium'
    else:
        return 'High', '#e84c4c', 'risk-high'


def get_value_flag(feature, value):
    meta = FEATURE_META[feature]
    if 'warn_high' in meta and value >= meta['warn_high']:
        return '<span class="metric-flag flag-high">↑ Above threshold</span>'
    if 'warn_low' in meta and value <= meta['warn_low']:
        return '<span class="metric-flag flag-low">↓ Below threshold</span>'
    return '<span class="metric-flag flag-normal">✓ Normal range</span>'


def make_gauge(prob, color):
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor='none')
    ax.set_facecolor('none')

    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), linewidth=18,
            color='#e8eaef', solid_capstyle='round')

    theta_fg = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(np.cos(theta_fg), np.sin(theta_fg), linewidth=18,
            color=color, solid_capstyle='round')

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate('', xy=(0.6 * np.cos(angle), 0.6 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#0f1f3d',
                                lw=2.5, mutation_scale=12))
    ax.plot(0, 0, 'o', color='#0f1f3d', markersize=8, zorder=5)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.25)
    ax.axis('off')

    for x_pos, label in [(-1.15, '0%'), (0, '50%'), (1.15, '100%')]:
        ax.text(x_pos, -0.15, label, ha='center', va='top',
                fontsize=8, color='#888', fontfamily='sans-serif')

    plt.tight_layout(pad=0)
    return fig


def make_feature_importance_chart(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()

    colors = ['#3a5fcd' if v >= importances.mean() else '#a8b8d0' for v in importances]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    bars = ax.barh(importances.index, importances.values,
                   color=colors, edgecolor='white', height=0.6)

    ax.axvline(importances.mean(), color='#0f1f3d', linestyle='--',
               alpha=0.4, linewidth=1, label='Mean importance')

    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8, color='#333')

    ax.set_xlabel('Importance Score', fontsize=9, color='#555')
    ax.set_title('What drives this model\'s predictions?', fontsize=10,
                 fontweight='600', color='#0f1f3d', pad=10)
    ax.tick_params(labelsize=9, colors='#444')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#ddd')
    ax.legend(fontsize=8, framealpha=0)
    plt.tight_layout()
    return fig


def make_patient_vs_population(input_values, df):
    features_to_show = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
    fig, axes = plt.subplots(1, len(features_to_show), figsize=(13, 3.5), facecolor='none')

    for ax, feat in zip(axes, features_to_show):
        non_diab = df[df['Outcome'] == 0][feat]
        diab     = df[df['Outcome'] == 1][feat]
        patient_val = input_values[feat]

        ax.hist(non_diab, bins=20, alpha=0.45, color='#3a9fd8', density=True)
        ax.hist(diab,     bins=20, alpha=0.45, color='#e84c4c', density=True)
        ax.axvline(patient_val, color='#0f1f3d', linewidth=2.5,
                   linestyle='-', zorder=5)
        ax.axvline(patient_val, color='#f5e642', linewidth=1,
                   linestyle='-', zorder=4)

        unit = FEATURE_META[feat].get('unit', '')
        ax.set_title(f'{feat}\n({patient_val:.1f} {unit})',
                     fontsize=8.5, fontweight='600', color='#0f1f3d')
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#ddd')
        ax.tick_params(labelsize=7.5, colors='#555')

    # Legend
    p1 = mpatches.Patch(color='#3a9fd8', alpha=0.6, label='Non-Diabetic')
    p2 = mpatches.Patch(color='#e84c4c', alpha=0.6, label='Diabetic')
    from matplotlib.lines import Line2D
    p3 = Line2D([0], [0], color='#0f1f3d', linewidth=2, label='Your values')
    fig.legend(handles=[p1, p2, p3], loc='lower center', ncol=3,
               fontsize=8, framealpha=0, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Your values vs. population distribution',
                 fontsize=10, fontweight='600', color='#0f1f3d', y=1.03)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────
model, medians = load_model()
df_pop = load_dataset()

# ─────────────────────────────────────────────
# Sidebar — patient input
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
            <div style='font-size:2.5rem;'>🩺</div>
            <div style='font-family: DM Serif Display, serif; font-size:1.4rem;
                        color: white; line-height:1.3;'>Diabetes Risk<br>Analyzer</div>
            <div style='font-size:0.75rem; color:#7a9abf; margin-top:0.3rem;'>
                Early Warning Signal Detection
            </div>
        </div>
        <hr style='border-color:#1e3560; margin: 0.8rem 0;'>
        <div style='font-size:0.8rem; color:#7a9abf; margin-bottom:1rem;
                    text-transform:uppercase; letter-spacing:0.08em; font-weight:600;'>
            Patient Clinical Values
        </div>
    """, unsafe_allow_html=True)

    input_values = {}
    for feat, meta in FEATURE_META.items():
        unit_label = f" ({meta['unit']})" if meta['unit'] else ''
        label = f"{meta['label']}{unit_label}"
        if meta['step'] == 1:
            val = st.slider(label, min_value=int(meta['min']),
                            max_value=int(meta['max']),
                            value=int(meta['default']), step=1)
        else:
            val = st.slider(label, min_value=float(meta['min']),
                            max_value=float(meta['max']),
                            value=float(meta['default']),
                            step=float(meta['step']))
        input_values[feat] = val

    st.markdown("<hr style='border-color:#1e3560; margin:1rem 0;'>", unsafe_allow_html=True)

    if st.button("🔄  Reset to median values", use_container_width=True):
        for feat, meta in FEATURE_META.items():
            input_values[feat] = meta['default']
        st.rerun()

    st.markdown("""
        <div style='font-size:0.72rem; color:#4a6a9a; margin-top:1rem; line-height:1.6;'>
            ⚠️ This tool is for educational purposes only. It is not a medical diagnostic device.
            Always consult a qualified healthcare professional.
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
input_df = pd.DataFrame([input_values])
prob      = model.predict_proba(input_df)[0][1]
pred      = model.predict(input_df)[0]
risk_level, risk_color, risk_class = get_risk_level(prob)

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='font-family: DM Serif Display, serif; color: #0f1f3d;
               font-size: 2.1rem; font-weight: 400; margin-bottom: 0.2rem;'>
        Diabetes Risk Early Warning System
    </h1>
    <p style='color:#666; font-size:0.95rem; margin-top:0;'>
        Adjust patient values in the sidebar · Results update in real time
    </p>
    <hr style='border-color:#e0e4ef; margin-bottom: 1.5rem;'>
""", unsafe_allow_html=True)

# ── Row 1: Risk card + gauge + input summary ──
col_result, col_gauge, col_inputs = st.columns([1.2, 1, 1.4])

with col_result:
    st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div style='font-size:0.8rem; font-weight:600; letter-spacing:0.1em;
                        text-transform:uppercase; color:#666; margin-bottom:0.3rem;'>
                Diabetes Risk Level
            </div>
            <div class="risk-label">{risk_level}</div>
            <div class="risk-prob">{prob:.1%}</div>
            <div class="risk-subtitle">
                {'Diabetes detected' if pred == 1 else 'No diabetes detected'}
                &nbsp;·&nbsp; Model confidence: {max(prob, 1-prob):.1%}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Risk thresholds explanation
    st.markdown("""
        <div style='font-size:0.78rem; color:#666; line-height:1.8;'>
            <span style='color:#34c97b; font-weight:600;'>● Low</span> &lt; 30% &nbsp;
            <span style='color:#f5a623; font-weight:600;'>● Moderate</span> 30–60% &nbsp;
            <span style='color:#e84c4c; font-weight:600;'>● High</span> &gt; 60%
        </div>
    """, unsafe_allow_html=True)

with col_gauge:
    st.markdown("<div style='text-align:center; font-size:0.8rem; color:#888; "
                "margin-bottom:0.3rem;'>Risk Probability Gauge</div>",
                unsafe_allow_html=True)
    fig_gauge = make_gauge(prob, risk_color)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

with col_inputs:
    st.markdown("<div class='section-header' style='margin-top:0;'>Patient Values</div>",
                unsafe_allow_html=True)
    for feat, val in input_values.items():
        unit = FEATURE_META[feat].get('unit', '')
        flag = get_value_flag(feat, val)
        display_val = f"{val:.1f} {unit}" if unit else str(val)
        st.markdown(f"""
            <div class="metric-tile">
                <div>
                    <div class="metric-name">{FEATURE_META[feat]['label']}</div>
                    <div class="metric-value">{display_val}</div>
                </div>
                {flag}
            </div>
        """, unsafe_allow_html=True)

# ── Row 2: Population comparison ──
st.markdown("<div class='section-header'>How do these values compare to the population?</div>",
            unsafe_allow_html=True)

if df_pop is not None:
    fig_pop = make_patient_vs_population(input_values, df_pop)
    st.pyplot(fig_pop, use_container_width=True)
    plt.close()
else:
    st.info("diabetes.csv not found — population comparison unavailable.")

# ── Row 3: Feature importance + clinical notes ──
col_imp, col_notes = st.columns([1.2, 1])

with col_imp:
    st.markdown("<div class='section-header'>Feature Importance</div>",
                unsafe_allow_html=True)
    fig_imp = make_feature_importance_chart(model, list(input_values.keys()))
    st.pyplot(fig_imp, use_container_width=True)
    plt.close()

with col_notes:
    st.markdown("<div class='section-header'>Clinical Notes</div>",
                unsafe_allow_html=True)

    # Flag features above warning thresholds
    flagged = []
    for feat, val in input_values.items():
        meta = FEATURE_META[feat]
        if 'warn_high' in meta and val >= meta['warn_high']:
            flagged.append(feat)

    if flagged:
        st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Elevated values detected</strong><br>
                The following indicators are above clinical thresholds:<br>
                <strong>{', '.join(flagged)}</strong>
            </div>
        """, unsafe_allow_html=True)

    for feat in flagged[:4]:  # show top 4 flagged
        st.markdown(f"""
            <div class="info-box">
                <strong>{FEATURE_META[feat]['label']}</strong><br>
                {FEATURE_DESCRIPTIONS[feat]}
            </div>
        """, unsafe_allow_html=True)

    if not flagged:
        st.markdown("""
            <div class="info-box">
                ✅ All entered values are within normal clinical ranges.
                No elevated risk indicators detected from individual feature thresholds.
            </div>
        """, unsafe_allow_html=True)

# ── Row 4: Model performance info ──
with st.expander("📊 About this model — performance & methodology"):
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    for col, metric, val in zip(
        [m_col1, m_col2, m_col3, m_col4, m_col5],
        ['Algorithm', 'Accuracy', 'ROC-AUC', 'Precision', 'Recall'],
        ['Random Forest', '~77%', '~0.82', '~73%', '~59%']
    ):
        col.metric(metric, val)

    st.markdown("""
    **Dataset:** Pima Indians Diabetes Database (768 patients, 8 clinical features)

    **Preprocessing:** Biologically impossible zero values in Glucose, BMI, BloodPressure,
    SkinThickness, and Insulin were replaced with column medians. Stratified 80/20 train-test split.

    **Model:** Random Forest Classifier (100 trees, max depth 5), selected over Logistic Regression,
    Gradient Boosting, and SVM based on cross-validated ROC-AUC.

    **Evaluation metric:** ROC-AUC is the primary metric (robust to class imbalance: 65%/35% split).

    **Limitations:** This model was trained on Pima Indian women aged 21+.
    Generalization to other populations should be done with caution.
    This tool does **not** replace clinical diagnosis.
    """)

st.markdown("""
    <hr style='border-color:#e0e4ef; margin-top:2rem;'>
    <div style='text-align:center; font-size:0.78rem; color:#aaa; padding-bottom:1rem;'>
        Week 7 — Data Science Project &nbsp;·&nbsp;
        Identifying Early Warning Signals of Diabetes Risk Using Routine Clinical Indicators
    </div>
""", unsafe_allow_html=True)