import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Human Voice Detection", layout="centered")
st.title("🎤 Human Voice Detection")
st.markdown("### Enter acoustic features below to instantly detect the speaker's gender")

# ============================= LOAD MODELS =============================
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.joblib')
    pca    = joblib.load('pca.joblib')
    svm    = joblib.load('svm_model.joblib')   
    kmeans = joblib.load('kmeans.joblib')
    return scaler, pca, svm, kmeans

scaler, pca, svm_model, kmeans = load_models()

# Get exact feature order from original dataset
@st.cache_data
def get_feature_order():
    df = pd.read_csv('vocal_gender_features_new.csv')
    return df.drop('label', axis=1).columns.tolist()

feature_names = get_feature_order()

# ============================= USER INPUT FORM =============================

with st.form(key="voice_form"):
    c1, c2 = st.columns(2)

    with c1:
        mean_pitch           = st.number_input("Mean Pitch (Hz)",          50.0,  500.0, 180.0)
        std_pitch            = st.number_input("Pitch Std Dev",            0.0,  300.0, 80.0)
        mean_centroid        = st.number_input("Mean Spectral Centroid",   500.0, 4000.0, 1800.0)
        rms_energy           = st.number_input("RMS Energy",               0.0,   1.0,   0.08)
        zcr                  = st.number_input("Zero Crossing Rate",       0.0,   0.3,   0.10)

    with c2:
        mfcc1 = st.number_input("MFCC 1 (mean)",  -1000.0, 100.0, -300.0)
        mfcc2 = st.number_input("MFCC 2 (mean)",   -200.0, 200.0,  80.0)
        mfcc3 = st.number_input("MFCC 3 (mean)",   -150.0, 150.0,  20.0)
        mfcc4 = st.number_input("MFCC 4 (mean)",   -100.0, 100.0, -10.0)
        mfcc5 = st.number_input("MFCC 5 (mean)",   -100.0, 100.0, -15.0)

    predict_btn = st.form_submit_button("🔮 Predict Gender")

# ============================= PREDICTION =============================
if predict_btn:
    features = np.zeros(len(feature_names))

    # Map the inputs we have
    mapping = {
        'mean_pitch'              : mean_pitch,
        'std_pitch'               : std_pitch,
        'mean_spectral_centroid'  : mean_centroid,
        'rms_energy'              : rms_energy,
        'zero_crossing_rate'      : zcr,
        'mfcc_1_mean'             : mfcc1,
        'mfcc_2_mean'             : mfcc2,
        'mfcc_3_mean'             : mfcc3,
        'mfcc_4_mean'             : mfcc4,
        'mfcc_5_mean'             : mfcc5,
    }

    for col_name, value in mapping.items():
        if col_name in feature_names:
            features[feature_names.index(col_name)] = value

    # Fill missing values with realistic column means from training data
    df_full = pd.read_csv('vocal_gender_features_new.csv').drop('label', axis=1)
    means = df_full.mean().values
    features = np.where(features == 0, means, features)

    # Reshape for sklearn
    X_input = features.reshape(1, -1)

    with st.spinner("Analyzing voice features..."):
        X_scaled = scaler.transform(X_input)
        X_pca    = pca.transform(X_scaled)

        gender_code = svm_model.predict(X_pca)[0]
        cluster     = kmeans.predict(X_pca)[0]

        gender = "Male Voice" if gender_code == 1 else "Female Voice"

    # ============================= RESULT DISPLAY =============================
    st.markdown("## 🎉 Result")

    if gender == "Male Voice":
        st.markdown("<h1 style='text-align:center; color:#1E88E5;'>🧔 MALE VOICE</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align:center; color:#EC407A;'>👩 FEMALE VOICE</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Gender", gender)
    col2.metric("Voice Cluster", cluster)

    with st.expander("See the PCA components that drove the decision"):
        pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        st.bar_chart(pca_df.T, use_container_width=True)

st.caption("Powered by SVM + PCA + K-Means — trained on thousands of real voice samples")