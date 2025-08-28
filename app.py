import streamlit as st
import joblib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf # For saving uploaded audio temporarily
import os
from scipy.stats import skew, kurtosis, entropy

# --- Streamlit UI ---
st.set_page_config(page_title="Voice Gender Predictor", layout="centered")

st.title("🗣️ Voice Gender Prediction")
st.markdown("Upload an audio file (WAV, MP3) to predict if the voice is male or female.")

# --- Configuration ---
SCALER_PATH = 'scaler.joblib'
PCA_PATH = 'pca.joblib'
MODEL_PATH = 'svm_model.joblib'

# Define the exact order of features as they were in your training data
# This is crucial for the scaler and PCA to work correctly on new data.

FEATURE_NAMES = [
    'mean_spectral_centroid', 'std_spectral_centroid', 'mean_spectral_bandwidth', 'std_spectral_bandwidth',
    'mean_spectral_contrast', 'mean_spectral_flatness', 'mean_spectral_rolloff', 'zero_crossing_rate',
    'rms_energy', 'mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch', 'spectral_skew', 'spectral_kurtosis',
    'energy_entropy', 'log_energy', 'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
    'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std',
    'mfcc_6_mean', 'mfcc_6_std', 'mfcc_7_mean', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std',
    'mfcc_9_mean', 'mfcc_9_std', 'mfcc_10_mean', 'mfcc_10_std', 'mfcc_11_mean', 'mfcc_11_std',
    'mfcc_12_mean', 'mfcc_12_std', 'mfcc_13_mean', 'mfcc_13_std'
]

# --- Load Pre-trained Models ---
@st.cache_resource # Cache the models to avoid reloading on every rerun
def load_models():
    """Loads the pre-trained StandardScaler, PCA, and classification model."""
    try:
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        model = joblib.load(MODEL_PATH)
        return scaler, pca, model
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure '{SCALER_PATH}', '{PCA_PATH}', and '{MODEL_PATH}' are in the same directory.")
        st.stop() # Stop the app if models can't be loaded
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

scaler, pca, model = load_models()

# --- Feature Extraction Function ---
def extract_audio_features(audio_file_path):
    """
    Extracts a predefined set of audio features from an audio file.
    Args:
        audio_file_path (str): Path to the audio file.
    Returns:
        np.array: A 2D numpy array containing the extracted features.
                  Returns None if feature extraction fails.
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=None) # Load with original sampling rate
        if len(y) == 0:
            st.error("Audio file is empty or could not be loaded.")
            return None

        features = {}

        # Spectral Features
        try:
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['mean_spectral_centroid'] = np.mean(cent)
            features['std_spectral_centroid'] = np.std(cent)
        except Exception as e:
            st.warning(f"Could not extract spectral centroid: {e}")
            features['mean_spectral_centroid'] = 0.0
            features['std_spectral_centroid'] = 0.0

        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['mean_spectral_bandwidth'] = np.mean(bandwidth)
            features['std_spectral_bandwidth'] = np.std(bandwidth)
        except Exception as e:
            st.warning(f"Could not extract spectral bandwidth: {e}")
            features['mean_spectral_bandwidth'] = 0.0
            features['std_spectral_bandwidth'] = 0.0

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['mean_spectral_contrast'] = np.mean(contrast)
        except Exception as e:
            st.warning(f"Could not extract spectral contrast: {e}")
            features['mean_spectral_contrast'] = 0.0

        try:
            flatness = librosa.feature.spectral_flatness(y=y)
            features['mean_spectral_flatness'] = np.mean(flatness)
        except Exception as e:
            st.warning(f"Could not extract spectral flatness: {e}")
            features['mean_spectral_flatness'] = 0.0

        try:
            rolloff = librosa.feature.spectral_rolloff(y=y)
            features['mean_spectral_rolloff'] = np.mean(rolloff)
        except Exception as e:
            st.warning(f"Could not extract spectral rolloff: {e}")
            features['mean_spectral_rolloff'] = 0.0

        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = np.mean(zcr)
        except Exception as e:
            st.warning(f"Could not extract zero crossing rate: {e}")
            features['zero_crossing_rate'] = 0.0

        try:
            rms = librosa.feature.rms(y=y)
            features['rms_energy'] = np.mean(rms)
        except Exception as e:
            st.warning(f"Could not extract RMS energy: {e}")
            features['rms_energy'] = 0.0

        # Pitch Features (using pyin for robustness)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_voiced = f0[voiced_flag.astype(bool)]
            if len(f0_voiced) > 0:
                features['mean_pitch'] = np.mean(f0_voiced)
                features['min_pitch'] = np.min(f0_voiced)
                features['max_pitch'] = np.max(f0_voiced)
                features['std_pitch'] = np.std(f0_voiced)
            else:
                st.warning("No voiced frames detected for pitch. Assigning default pitch values.")
                features['mean_pitch'] = 0.0
                features['min_pitch'] = 0.0
                features['max_pitch'] = 0.0
                features['std_pitch'] = 0.0
        except Exception as e:
            st.warning(f"Could not extract pitch features: {e}. Assigning default pitch values.")
            features['mean_pitch'] = 0.0
            features['min_pitch'] = 0.0
            features['max_pitch'] = 0.0
            features['std_pitch'] = 0.0

        # Spectral Skewness and Kurtosis (from magnitude spectrum)
        try:
            S = np.abs(librosa.stft(y))
            S_mean = np.mean(S, axis=1)
            if len(S_mean) > 1: # Ensure enough data points for skew/kurtosis
                features['spectral_skew'] = skew(S_mean)
                features['spectral_kurtosis'] = kurtosis(S_mean)
            else:
                st.warning("Not enough data for spectral skew/kurtosis. Assigning default values.")
                features['spectral_skew'] = 0.0
                features['spectral_kurtosis'] = 0.0
        except Exception as e:
            st.warning(f"Could not extract spectral skew/kurtosis: {e}. Assigning default values.")
            features['spectral_skew'] = 0.0
            features['spectral_kurtosis'] = 0.0

        # Energy Entropy and Log Energy
        try:
            frame_length = 2048
            hop_length = 512
            frame_energies = np.array([
                np.sum(y[i:i+frame_length]**2)
                for i in range(0, len(y) - frame_length, hop_length)
            ])
            if np.sum(frame_energies) > 0:
                normalized_energies = frame_energies / np.sum(frame_energies)
                features['energy_entropy'] = entropy(normalized_energies)
            else:
                st.warning("No energy detected for entropy. Assigning default value.")
                features['energy_entropy'] = 0.0

            # Log Energy: log of mean squared RMS energy
            # Ensure rms is defined from previous block, if not, re-calculate or use default
            if 'rms' not in locals() or rms is None or np.mean(rms**2) <= 0:
                st.warning("RMS energy not available or zero for log energy. Assigning default value.")
                features['log_energy'] = np.log10(1e-10) # Smallest possible log value
            else:
                features['log_energy'] = np.log10(np.mean(rms**2)) # No 1e-10 here if rms is valid
        except Exception as e:
            st.warning(f"Could not extract energy entropy/log energy: {e}. Assigning default values.")
            features['energy_entropy'] = 0.0
            features['log_energy'] = np.log10(1e-10)


        # MFCCs
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(1, 14):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i-1])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i-1])
        except Exception as e:
            st.warning(f"Could not extract MFCCs: {e}. Assigning default values for all MFCCs.")
            for i in range(1, 14):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0


        # Convert dictionary to Pandas Series and then to a 2D array,
        # ensuring the column order matches the training data
        extracted_series = pd.Series(features)
        extracted_series = extracted_series.reindex(FEATURE_NAMES) # Reindex to ensure order

        # Check for any NaN values introduced during feature extraction for debugging
        if extracted_series.isnull().any():
            st.error("NaN values detected in extracted features. Please check audio file or feature extraction logic.")
            st.write("Features with NaN:", extracted_series[extracted_series.isnull()])
            return None

        return extracted_series.values.reshape(1, -1)

    except Exception as e:
        st.error(f"An unhandled error occurred during audio loading or initial feature extraction: {e}")
        return None

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=uploaded_file.type)

    # Create a temporary file to save the uploaded audio
    temp_audio_path = "temp_audio_file." + uploaded_file.name.split('.')[-1]
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        temp_audio_path = None # Indicate that temp file creation failed

    if temp_audio_path: # Proceed only if temp file was successfully created
        st.write("Processing audio...")

        # Extract features
        with st.spinner("Extracting audio features..."):
            features_extracted = extract_audio_features(temp_audio_path)

        if features_extracted is not None:
            st.success("Features extracted successfully!")

            # Preprocess features
            with st.spinner("Scaling and applying PCA..."):
                try:
                    features_scaled = scaler.transform(features_extracted)
                    features_pca = pca.transform(features_scaled)
                except Exception as e:
                    st.error(f"Error during scaling or PCA transformation: {e}")
                    features_pca = None # Indicate preprocessing failed

            if features_pca is not None:
                st.success("Features preprocessed!")

                # Make prediction
                with st.spinner("Making prediction..."):
                    try:
                        prediction = model.predict(features_pca)
                        prediction_proba = model.predict_proba(features_pca)

                        gender_label = "Female" if prediction[0] == 0 else "Male"
                        confidence = prediction_proba[0][prediction[0]] * 100

                        st.subheader(f"Predicted Gender: **{gender_label}**")
                        st.write(f"Confidence: **{confidence:.2f}%**")

                        st.markdown("---")
                        st.subheader("How it works:")
                        st.markdown(
                            """
                            This application uses a pre-trained machine learning model to classify voice samples as male or female.
                            It extracts various audio features (like spectral characteristics, pitch, and MFCCs),
                            then scales and reduces their dimensionality using PCA, and finally feeds them into a classification model
                            (e.g., Random Forest or SVM) for prediction.
                            """
                        )
                    except Exception as e:
                        st.error(f"Error during model prediction: {e}")
            else:
                st.error("Feature preprocessing failed. Cannot make a prediction.")
        else:
            st.error("Could not extract features from the audio file. Please try another file.")

    # Clean up the temporary file
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)